"""Run inference with the base Llama-3 (or smaller) model before fine-tuning.
Usage: python before.py "<prompt>"
"""

import sys
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Hello, how are you?"
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
    messages = f"{tokenizer.bos_token}<|user|>{prompt}<|assistant|>"
    out = pipe(messages, max_new_tokens=256, do_sample=True, temperature=0.7)
    print(out[0]["generated_text"][len(messages):].strip())

if __name__ == "__main__":
    main()
