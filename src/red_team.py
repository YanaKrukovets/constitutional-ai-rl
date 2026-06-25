"""
The red-teaming loop: the differentiator that shows the model getting more
robust against jailbreak attempts across rounds, not just better on a static
eval set.

For each round:
  1. Run every adversarial prompt through the current model, judge each
     response on its own (Judge.classify), and record the jailbreak success
     rate (fraction of adversarial prompts where the model violated >=1 rule).
  2. For every prompt that broke a rule, try to find a safe alternative by
     resampling the same model a few times (rejection sampling) and keeping
     the first resample the judge says is clean. If none of the resamples are
     clean, fall back to a templated refusal that cites the violated rule, so
     every failure still produces a usable (chosen, rejected) pair.
  3. Write those pairs to data/red_team_pairs_round{N}.jsonl and run another
     DPO round *continuing from the previous round's adapter* (train_dpo.py
     --init-adapter), producing outputs/red_team_round{N+1}.
  4. Move to the next round using the newly trained adapter.

Writes results/red_team_metrics.json (success rate per round) and a plot at
results/plots/red_team_success_rate.png.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
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
OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
RESAMPLE_TRIES = 3


def load_adversarial_prompts():
    prompts = []
    for filename in ("prompts_train.jsonl", "prompts_eval.jsonl"):
        with open(DATA_DIR / filename) as f:
            for line in f:
                r = json.loads(line)
                if r["bucket"] == "adversarial":
                    prompts.append(r)
    return prompts


def fallback_refusal(rule_id: str, rules_by_id: dict) -> str:
    rule = rules_by_id.get(rule_id)
    if rule is None:
        return "I can't help with that request because it would violate my guidelines."
    return (
        f"I can't help with that request because it would go against one of my "
        f"rules: \"{rule['title']}\". {rule['why']}"
    )


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_round(model, tokenizer, device, judge, attack_records, rules_by_id, max_new_tokens):
    """Generates a response for every attack prompt, classifies it, and
    returns (success_rate, list_of_failure_records)."""
    failures = []
    n_violations = 0
    for r in attack_records:
        response = generate(tokenizer, model, device, r["text"], max_new_tokens, temperature=0.0)
        verdict = judge.classify(r["text"], response)
        if verdict["violated_rules"]:
            n_violations += 1
            failures.append({"prompt": r["text"], "response": response, "violated_rules": verdict["violated_rules"]})
    success_rate = n_violations / len(attack_records) if attack_records else 0.0
    return success_rate, failures


def build_round_pairs(model, tokenizer, device, judge, failures, rules_by_id, max_new_tokens):
    """For each failed (jailbroken) prompt, find a safe chosen response via
    rejection sampling, falling back to a templated refusal if resampling
    doesn't produce a clean one."""
    pairs = []
    for f in failures:
        chosen = None
        for _ in range(RESAMPLE_TRIES):
            candidate = generate(tokenizer, model, device, f["prompt"], max_new_tokens, temperature=0.9)
            candidate_verdict = judge.classify(f["prompt"], candidate)
            if not candidate_verdict["violated_rules"]:
                chosen = candidate
                break
        synthetic = chosen is None
        if chosen is None:
            chosen = fallback_refusal(f["violated_rules"][0], rules_by_id)

        pairs.append({
            "prompt": f["prompt"],
            "chosen": chosen,
            "rejected": f["response"],
            "violated_rules_rejected": f["violated_rules"],
            "synthetic_chosen": synthetic,
        })
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--base-model", default=DEFAULT_MODEL)
    parser.add_argument("--init-adapter", default=str(OUTPUTS_DIR / "dpo_adapter"),
                         help="adapter to start red-teaming from (default: the round-1 DPO adapter from train_dpo.py)")
    parser.add_argument("--judge-backend", default=None)
    parser.add_argument("--judge-device", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=150)
    parser.add_argument("--train-epochs", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None, help="only use first N adversarial prompts, for quick smoke tests")
    args = parser.parse_args()

    attack_records = load_adversarial_prompts()
    if args.limit:
        attack_records = attack_records[: args.limit]
    print(f"Loaded {len(attack_records)} adversarial prompts for red-teaming")

    rules_by_id = {r["id"]: r for r in load_rules()}
    judge = Judge(backend=args.judge_backend, device=args.judge_device)
    print(f"Judge backend: {judge.backend} ({judge.model})")

    device = args.device or pick_device()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model)
    base_model.to(device)

    current_adapter_dir = args.init_adapter
    print(f"Starting red-teaming from adapter: {current_adapter_dir}")

    round_metrics = []

    for round_idx in range(args.rounds):
        model = PeftModel.from_pretrained(base_model, current_adapter_dir)
        model.to(device)
        model.eval()

        success_rate, failures = run_round(
            model, tokenizer, device, judge, attack_records, rules_by_id, args.max_new_tokens
        )
        print(f"[round {round_idx}] jailbreak success rate: {success_rate:.2%} "
              f"({len(failures)}/{len(attack_records)} prompts broke a rule)")
        round_metrics.append({
            "round": round_idx,
            "jailbreak_success_rate": success_rate,
            "n_failures": len(failures),
            "n_attack_prompts": len(attack_records),
            "adapter_dir": current_adapter_dir,
        })

        is_last_round = round_idx == args.rounds - 1
        if is_last_round or not failures:
            break

        pairs = build_round_pairs(model, tokenizer, device, judge, failures, rules_by_id, args.max_new_tokens)
        pairs_path = DATA_DIR / f"red_team_pairs_round{round_idx}.jsonl"
        with open(pairs_path, "w") as f:
            for p in pairs:
                f.write(json.dumps(p) + "\n")
        print(f"[round {round_idx}] wrote {len(pairs)} retraining pairs to {pairs_path}")

        next_adapter_dir = OUTPUTS_DIR / f"red_team_round{round_idx + 1}"
        del model  # free the PeftModel wrapping the shared base_model before the training subprocess loads its own copy
        train_cmd = [
            sys.executable, str(Path(__file__).parent / "train_dpo.py"),
            "--preference-data", str(pairs_path),
            "--base-model", args.base_model,
            "--init-adapter", current_adapter_dir,
            "--output-dir", str(next_adapter_dir),
            "--epochs", str(args.train_epochs),
            "--batch-size", "1",
        ]
        if device == "cpu":
            train_cmd.append("--force-cpu")
        print(f"[round {round_idx}] retraining: {' '.join(train_cmd)}")
        subprocess.run(train_cmd, check=True)

        current_adapter_dir = str(next_adapter_dir)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "red_team_metrics.json", "w") as f:
        json.dump(round_metrics, f, indent=2)
    print("\n=== Red-team summary ===")
    print(json.dumps(round_metrics, indent=2))

    try:
        plot_success_rate(round_metrics)
    except Exception as e:
        print(f"Skipping plot generation ({e})")


def plot_success_rate(round_metrics: list):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = RESULTS_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    rounds = [m["round"] for m in round_metrics]
    rates = [m["jailbreak_success_rate"] for m in round_metrics]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(rounds, rates, marker="o")
    ax.set_xlabel("red-team round")
    ax.set_ylabel("jailbreak success rate")
    ax.set_title("Jailbreak success rate across red-team rounds")
    ax.set_ylim(0, 1)
    ax.set_xticks(rounds)
    fig.tight_layout()
    fig.savefig(plots_dir / "red_team_success_rate.png", dpi=150)
    print(f"Saved plot to {plots_dir / 'red_team_success_rate.png'}")


if __name__ == "__main__":
    main()
