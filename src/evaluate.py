"""
Evaluates the DPO-tuned model against the untouched base model on the
held-out eval prompts (never seen during training/judging in step 5).

For every eval prompt:
  - generate a response from the base model
  - generate a response from base model + LoRA adapter ("tuned")
  - ask the judge to compare them (same judge used to build training pairs)

Reports:
  - overall violation rate, base vs tuned
  - per-rule violation rate, base vs tuned (the chart that matters most)
  - per-bucket violation rate (benign / boundary / adversarial)
  - win-rate of tuned over base in head-to-head judge comparisons
  - average response length, base vs tuned (sanity check against the
    over-refusal failure mode: tuned model becoming uselessly evasive,
    which would show up as much shorter, vaguer responses on benign prompts)

Writes results/metrics.json and a bar chart to results/plots/.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch

if not hasattr(torch.backends.mps, "is_macos_or_newer"):
    torch.backends.mps.is_macos_or_newer = lambda *args, **kwargs: True

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from constitution import load_rules
from judge import Judge
from sample_responses import generate

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-prompts", default=str(DATA_DIR / "prompts_eval.jsonl"))
    parser.add_argument("--base-model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter-dir", default=str(Path(__file__).resolve().parent.parent / "outputs" / "dpo_adapter"))
    parser.add_argument("--judge-backend", default=None)
    parser.add_argument("--judge-device", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--device", default=None, help="force a device for the policy models, e.g. 'cpu'")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    with open(args.eval_prompts) as f:
        eval_records = [json.loads(line) for line in f]
    if args.limit:
        eval_records = eval_records[: args.limit]

    device = args.device or pick_device()
    print(f"Loading base model {args.base_model} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model)
    base_model.to(device)
    base_model.eval()

    print(f"Loading LoRA adapter from {args.adapter_dir} ...")
    tuned_model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    tuned_model.to(device)
    tuned_model.eval()

    judge = Judge(backend=args.judge_backend, device=args.judge_device)
    print(f"Judge backend: {judge.backend} ({judge.model})")

    rule_ids = [r["id"] for r in load_rules()]
    violations_base_by_rule = Counter()
    violations_tuned_by_rule = Counter()
    violations_base_by_bucket = defaultdict(int)
    violations_tuned_by_bucket = defaultdict(int)
    count_by_bucket = defaultdict(int)
    tuned_wins = 0
    total = 0
    base_lengths, tuned_lengths = [], []

    per_example = []

    for i, r in enumerate(eval_records):
        prompt = r["text"]
        base_response = generate(tokenizer, base_model, device, prompt, args.max_new_tokens, temperature=0.0)
        tuned_response = generate(tokenizer, tuned_model, device, prompt, args.max_new_tokens, temperature=0.0)

        verdict = judge.compare(prompt, base_response, tuned_response)
        # A = base, B = tuned, per the call order above
        base_violations = verdict["violated_rules_a"]
        tuned_violations = verdict["violated_rules_b"]

        for rule_id in base_violations:
            violations_base_by_rule[rule_id] += 1
        for rule_id in tuned_violations:
            violations_tuned_by_rule[rule_id] += 1

        if base_violations:
            violations_base_by_bucket[r["bucket"]] += 1
        if tuned_violations:
            violations_tuned_by_bucket[r["bucket"]] += 1
        count_by_bucket[r["bucket"]] += 1

        if verdict["preferred"] == "B":
            tuned_wins += 1
        total += 1

        base_lengths.append(len(base_response.split()))
        tuned_lengths.append(len(tuned_response.split()))

        per_example.append({
            "prompt": prompt,
            "bucket": r["bucket"],
            "base_response": base_response,
            "tuned_response": tuned_response,
            "base_violations": base_violations,
            "tuned_violations": tuned_violations,
            "preferred": verdict["preferred"],
            "rationale": verdict["rationale"],
        })
        print(f"[{i + 1}/{len(eval_records)}] preferred={verdict['preferred']} "
              f"base_violations={base_violations} tuned_violations={tuned_violations}")

    metrics = {
        "n_eval_prompts": total,
        "overall": {
            "base_violation_rate": sum(violations_base_by_rule.values()) / total if total else 0.0,
            "tuned_violation_rate": sum(violations_tuned_by_rule.values()) / total if total else 0.0,
            "win_rate_tuned_over_base": tuned_wins / total if total else 0.0,
            "avg_response_length_base": sum(base_lengths) / total if total else 0.0,
            "avg_response_length_tuned": sum(tuned_lengths) / total if total else 0.0,
        },
        "per_rule": {
            rule_id: {
                "base_violations": violations_base_by_rule.get(rule_id, 0),
                "tuned_violations": violations_tuned_by_rule.get(rule_id, 0),
            }
            for rule_id in rule_ids
        },
        "per_bucket": {
            bucket: {
                "n": count_by_bucket[bucket],
                "base_violation_rate": violations_base_by_bucket[bucket] / count_by_bucket[bucket],
                "tuned_violation_rate": violations_tuned_by_bucket[bucket] / count_by_bucket[bucket],
            }
            for bucket in count_by_bucket
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(RESULTS_DIR / "eval_examples.json", "w") as f:
        json.dump(per_example, f, indent=2)

    print("\n=== Summary ===")
    print(json.dumps(metrics["overall"], indent=2))
    print(json.dumps(metrics["per_rule"], indent=2))

    try:
        plot_results(metrics)
    except Exception as e:
        print(f"Skipping plot generation ({e})")


def plot_results(metrics: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = RESULTS_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    rule_ids = list(metrics["per_rule"].keys())
    base_counts = [metrics["per_rule"][r]["base_violations"] for r in rule_ids]
    tuned_counts = [metrics["per_rule"][r]["tuned_violations"] for r in rule_ids]

    x = range(len(rule_ids))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([i - width / 2 for i in x], base_counts, width, label="base model")
    ax.bar([i + width / 2 for i in x], tuned_counts, width, label="DPO-tuned model")
    ax.set_xticks(list(x))
    ax.set_xticklabels(rule_ids, rotation=45, ha="right")
    ax.set_ylabel("violation count")
    ax.set_title("Constitution violations by rule: base vs. DPO-tuned")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "violations_by_rule.png", dpi=150)
    print(f"Saved plot to {plots_dir / 'violations_by_rule.png'}")


if __name__ == "__main__":
    main()
