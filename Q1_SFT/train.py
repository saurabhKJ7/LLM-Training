"""Fine-tune Llama-3 (or smaller) with LoRA using PEFT on the custom dataset.

Requirements:
    pip install transformers datasets peft accelerate bitsandbytes

Run:
    python train.py --model NousResearch/Meta-Llama-3-8B-Instruct --epochs 3
"""
import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                  help="Model to fine-tune (default: gpt2)")
    p.add_argument("--data", default="dataset.json",
                  help="Path to training data JSON file")
    p.add_argument("--batch_size", type=int, default=1,
                  help="Batch size (default: 1)")
    p.add_argument("--epochs", type=int, default=5,
                  help="Number of training epochs (default: 3)")
    p.add_argument("--lr", type=float, default=5e-5,
                  help="Learning rate (default: 2e-4)")
    p.add_argument("--output_dir", default="lora-out",
                  help="Directory to save the fine-tuned model")
    return p.parse_args()


def main():
    args = parse_args()

    # Load tokenizer and base model
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token

    print("Loading dataset…")
    dataset = load_dataset("json", data_files=args.data, split="train")

    def tokenize_fn(example):
        return tokenizer(example["text"], truncation=True, max_length=512)

    # Wrap with BOS/EOS tokens (TinyLlama still uses the same special tokens)
    def add_special(example):
        example["text"] = tokenizer.bos_token + example["text"] + tokenizer.eos_token
        return example

    dataset = dataset.map(add_special)
    tokenized = dataset.map(tokenize_fn, remove_columns=["text"], batched=True)

    # Load model without quantization
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        use_cache=False
    )
    model.gradient_checkpointing_enable()  # Enable gradient checkpointing

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        save_strategy="epoch",
        gradient_accumulation_steps=4,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    print("Starting training…")
    trainer.train()

    print("Saving LoRA adapter to", args.output_dir)
    trainer.save_model(args.output_dir)

    print("Done.")

if __name__ == "__main__":
    main()
