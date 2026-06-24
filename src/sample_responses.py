"""
Loads the base model once and, for every prompt in a prompts jsonl file,
samples 2 completions at temperature ~0.9 (cheap diversity trick: same model,
different random sampling, no need to spin up multiple model copies).

Writes a new jsonl where each line is:
  {"id", "text", "bucket", "rule", "response_a", "response_b"}

This output feeds straight into judge.py, which picks which of A/B better
follows the constitution.
"""

import argparse
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def load_model(model_name):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
    )
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model, device


def generate(tokenizer, model, device, prompt_text, max_new_tokens, temperature):
    import torch

    messages = [{"role": "user", "content": prompt_text}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        output = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0][inputs.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DATA_DIR / "prompts_train.jsonl"))
    parser.add_argument("--output", default=str(DATA_DIR / "sampled_responses.jsonl"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--limit", type=int, default=None, help="only process first N prompts (for quick smoke tests)")
    args = parser.parse_args()

    with open(args.input) as f:
        records = [json.loads(line) for line in f]
    if args.limit:
        records = records[: args.limit]

    print(f"Loading model {args.model} ...")
    tokenizer, model, device = load_model(args.model)
    print(f"Model loaded on device: {device}")

    out_path = Path(args.output)
    with open(out_path, "w") as f:
        for i, r in enumerate(records):
            response_a = generate(tokenizer, model, device, r["text"], args.max_new_tokens, args.temperature)
            response_b = generate(tokenizer, model, device, r["text"], args.max_new_tokens, args.temperature)
            r["response_a"] = response_a
            r["response_b"] = response_b
            f.write(json.dumps(r) + "\n")
            print(f"[{i + 1}/{len(records)}] sampled 2 responses for: {r['text'][:60]}...")

    print(f"Wrote {len(records)} records with sampled responses to {out_path}")


if __name__ == "__main__":
    main()
