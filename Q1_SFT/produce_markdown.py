"""Generate a before_after.md comparing base TinyLlama and the LoRA-tuned adapter on a few reference prompts.

Run:
    python produce_markdown.py --adapter_dir lora-out
"""
import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import AutoPeftModelForCausalLM

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
PROMPTS = [
    "What is the capital of France?",
    "Please explain photosynthesis in one sentence.",
    "Could you politely refuse to share my personal password?",
    "Give me a short (max 20 words) benefit of daily exercise.",
    "Tell a fun fact about cats."
]

def load_pipelines(adapter_dir):
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto")
    base_pipe = pipeline("text-generation", model=base_model, tokenizer=tok, device_map="auto")

    tuned_model = AutoPeftModelForCausalLM.from_pretrained(adapter_dir)
    if torch.backends.mps.is_available():
        tuned_model.to("mps")
    tuned_pipe = pipeline("text-generation", model=tuned_model, tokenizer=tok, device_map=None)
    return tok, base_pipe, tuned_pipe

def run_comparison(tok, base_pipe, tuned_pipe, prompt, max_new=64):
    bos = tok.bos_token
    text_in = f"{bos}<|user|>{prompt}<|assistant|>"
    base_out = base_pipe(text_in, max_new_tokens=max_new, do_sample=False)[0]["generated_text"][len(text_in):].strip()
    tuned_out = tuned_pipe(text_in, max_new_tokens=max_new, do_sample=False)[0]["generated_text"][len(text_in):].strip()
    return base_out, tuned_out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter_dir", default="lora-out")
    args = ap.parse_args()

    tok, base_pipe, tuned_pipe = load_pipelines(args.adapter_dir)

    lines = ["# Before vs After Fine-Tuning\n"]
    for p in PROMPTS:
        base, tuned = run_comparison(tok, base_pipe, tuned_pipe, p)
        lines.append(f"## Prompt\n`{p}`\n")
        lines.append("<details><summary>Base TinyLlama</summary>\n\n" + base + "\n</details>\n")
        lines.append("<details><summary>Fine-tuned (LoRA)</summary>\n\n" + tuned + "\n</details>\n")
    Path("before_after.md").write_text("\n".join(lines))
    print("Written before_after.md")

if __name__ == "__main__":
    main()
