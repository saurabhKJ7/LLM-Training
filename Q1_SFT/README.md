# Polite Helper Fine-Tuning Project

This project demonstrates how to fine-tune a base language model (TinyLlama-1.1B-Chat) using LoRA (Low-Rank Adaptation) and a custom dataset to create a polite, helpful, and refusal-capable AI assistant.

## Project Structure

- `dataset.json` — Custom dataset with prompt-response pairs (polite, factual, refusal, etc.)
- `train.py` — Fine-tunes TinyLlama with LoRA using PEFT
- `before.py` — Runs inference on the base model (before fine-tuning)
- `compare.py` — Compares base model vs. fine-tuned responses
- `produce_markdown.py` — Generates `before_after.md` with side-by-side outputs
- `lora-out/` — Directory containing LoRA adapter weights after training
- `before_after.md` — Markdown summary of model outputs before and after fine-tuning

## How to Run

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the model:**
   ```bash
   python train.py --epochs 3 --batch_size 1
   ```

3. **Compare responses:**
   ```bash
   python compare.py "What is the capital of France?" --adapter_dir lora-out
   ```

4. **Generate a markdown report:**
   ```bash
   python produce_markdown.py --adapter_dir lora-out
   # View before_after.md for results
   ```

## Example Results
See `before_after.md` for side-by-side outputs of the base and fine-tuned models.

## Key Features
- Uses LoRA for efficient fine-tuning
- Polite, factual, and refusal behaviors
- Works on Mac (M1/M2) with MPS acceleration
- Easily extensible: add more data to `dataset.json` for improved results

## Tips
- For best results, expand `dataset.json` with more examples (especially for factual Q&A)
- You can change the base model in `train.py` and `before.py` (e.g., to try other instruction-tuned LLMs)
- Use `compare.py` and `before.py` to evaluate model behavior interactively

---

*Created by Saurabh Kumar Jha (2025).*
