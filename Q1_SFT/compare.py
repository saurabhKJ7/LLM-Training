"""Compare responses from base GPT-2 vs. your LoRA-fine-tuned adapter.

Usage:
    python compare.py "<prompt>" --adapter_dir lora-out

Dependencies: transformers, peft
"""
import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import AutoPeftModelForCausalLM

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Must match the model used in training

def build_pipeline(model_name_or_path):
    """Utility to build a text-generation pipeline."""
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto")
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
    )

def build_adapter_pipeline(adapter_dir):
    model = AutoPeftModelForCausalLM.from_pretrained(adapter_dir)
    if torch.backends.mps.is_available():
        model.to("mps")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    return pipeline("text-generation", model=model, tokenizer=tokenizer, device_map=None)

def main():
    parser = argparse.ArgumentParser(description="Compare base vs fine-tuned model responses.")
    parser.add_argument("prompt", help="Prompt to evaluate")
    parser.add_argument("--adapter_dir", default="lora-out", help="Directory with LoRA adapter weights")
    parser.add_argument("--max_tokens", type=int, default=128)
    args = parser.parse_args()

    prompt = args.prompt
    bos = AutoTokenizer.from_pretrained(BASE_MODEL).bos_token
    user_prompt = f"{bos}<|user|>{prompt}<|assistant|>"

    print("Loading base model…")
    base_pipe = build_pipeline(BASE_MODEL)

    print("Loading fine-tuned adapter…")
    tuned_pipe = build_adapter_pipeline(args.adapter_dir)

    print("\n=== Base GPT-2 ===")
    base_out = base_pipe(user_prompt, max_new_tokens=args.max_tokens, do_sample=True, temperature=0.7)[0]["generated_text"][len(user_prompt):].strip()
    print(base_out)

    print("\n=== Fine-tuned (LoRA) ===")
    tuned_out = tuned_pipe(user_prompt, max_new_tokens=args.max_tokens, do_sample=True, temperature=0.7)[0]["generated_text"][len(user_prompt):].strip()
    print(tuned_out)

if __name__ == "__main__":
    main()
