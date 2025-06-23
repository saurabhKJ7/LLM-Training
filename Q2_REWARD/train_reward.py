"""
Train a reward model using HuggingFace TRL's RewardTrainer on the answers.csv file.
The reward model learns to predict higher scores for preferred answers (rank 1) and
lower scores for less preferred answers (rank 4).

Usage
-----
$ python train_reward.py --csv_path answers.csv --output_dir reward_model
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments
from trl import RewardTrainer, RewardConfig

MODEL_NAME = "distilbert-base-uncased"  # small & fast encoder model


def load_dataset(csv_path: Path):
    df = pd.read_csv(csv_path)
    # Convert rank -> reward score (invert so 1 -> 4)
    df["reward"] = df["rank"].apply(lambda r: 5 - r)  # 4,3,2,1
    # Combine prompt & answer into a single text input
    df["text"] = df.apply(lambda row: row["prompt"] + " " + row["answer"], axis=1)
    return Dataset.from_pandas(df[["text", "reward"]])


def main(csv_path: Path, output_dir: Path, num_train_steps: int = 80):
    dataset = load_dataset(csv_path)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1)

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)

    tokenized = dataset.map(tokenize_fn, batched=True)

    # TRL expects the label column to be called "rewards"
    tokenized = tokenized.rename_column("reward", "rewards")

    args = TrainingArguments(
        per_device_train_batch_size=4,
        num_train_epochs=1,  # we'll rely on max_steps
        max_steps=num_train_steps,
        learning_rate=2e-5,
        logging_steps=10,
        evaluation_strategy="no",
        output_dir=output_dir,
        save_total_limit=1,
    )

    reward_config = RewardConfig(training_arguments=args)
    trainer = RewardTrainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
        reward_config=reward_config,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=Path, default=Path(__file__).parent / "answers.csv")
    parser.add_argument("--output_dir", type=Path, default=Path(__file__).parent / "reward_model")
    parser.add_argument("--steps", type=int, default=80)
    args = parser.parse_args()
    main(args.csv_path, args.output_dir, args.steps)
