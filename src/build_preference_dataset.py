"""
Orchestrates the full preference-pair pipeline:

  prompts_train.jsonl -> sample 2 responses per prompt (sample_responses.py)
                       -> judge picks the better one (judge.py)
                       -> writes {prompt, chosen, rejected} pairs for DPO

Also logs the pre-training ("baseline") per-rule violation counts, since that's
the "before" half of the before/after metric reported in evaluate.py later.

Output:
  data/preference_pairs.jsonl   - {prompt, chosen, rejected, bucket, rule,
                                    violated_rules_chosen, violated_rules_rejected}
  data/baseline_violations.json - per-rule violation counts across all sampled
                                   responses (both A and B), before any training
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from judge import Judge
from sample_responses import DEFAULT_MODEL, generate, load_model

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DATA_DIR / "prompts_train.jsonl"))
    parser.add_argument("--output", default=str(DATA_DIR / "preference_pairs.jsonl"))
    parser.add_argument("--baseline-output", default=str(DATA_DIR / "baseline_violations.json"))
    parser.add_argument("--policy-model", default=DEFAULT_MODEL)
    parser.add_argument("--judge-backend", default=None, help="'anthropic' or 'local'; default auto-detects API key")
    parser.add_argument("--judge-device", default=None, help="force judge model device, e.g. 'cpu' (useful when policy + judge models don't both fit on a small GPU)")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    with open(args.input) as f:
        records = [json.loads(line) for line in f]
    if args.limit:
        records = records[: args.limit]

    print(f"Loading policy model {args.policy_model} ...")
    tokenizer, model, device = load_model(args.policy_model)
    print(f"Policy model loaded on device: {device}")

    judge = Judge(backend=args.judge_backend, device=args.judge_device)
    print(f"Judge backend: {judge.backend} ({judge.model})")

    rule_violation_counts = Counter()
    total_responses_judged = 0
    pairs_written = 0

    out_path = Path(args.output)
    with open(out_path, "w") as f:
        for i, r in enumerate(records):
            response_a = generate(tokenizer, model, device, r["text"], args.max_new_tokens, args.temperature)
            response_b = generate(tokenizer, model, device, r["text"], args.max_new_tokens, args.temperature)

            verdict = judge.compare(r["text"], response_a, response_b)

            for rule_id in verdict["violated_rules_a"]:
                rule_violation_counts[rule_id] += 1
            for rule_id in verdict["violated_rules_b"]:
                rule_violation_counts[rule_id] += 1
            total_responses_judged += 2

            if verdict["preferred"] == "A":
                chosen, rejected = response_a, response_b
                violated_chosen, violated_rejected = verdict["violated_rules_a"], verdict["violated_rules_b"]
            else:
                chosen, rejected = response_b, response_a
                violated_chosen, violated_rejected = verdict["violated_rules_b"], verdict["violated_rules_a"]

            # skip pairs where chosen == rejected text (judge had nothing to discriminate on)
            if chosen.strip() == rejected.strip():
                print(f"[{i + 1}/{len(records)}] skipped (identical responses): {r['text'][:60]}...")
                continue

            pair = {
                "prompt": r["text"],
                "chosen": chosen,
                "rejected": rejected,
                "bucket": r["bucket"],
                "rule": r["rule"],
                "violated_rules_chosen": violated_chosen,
                "violated_rules_rejected": violated_rejected,
                "rationale": verdict["rationale"],
            }
            f.write(json.dumps(pair) + "\n")
            pairs_written += 1
            print(f"[{i + 1}/{len(records)}] preferred={verdict['preferred']} "
                  f"violations(A={verdict['violated_rules_a']}, B={verdict['violated_rules_b']}): "
                  f"{r['text'][:60]}...")

    baseline = {
        "total_responses_judged": total_responses_judged,
        "violation_count_by_rule": dict(rule_violation_counts),
        "violation_rate_by_rule": {
            rule: count / total_responses_judged
            for rule, count in rule_violation_counts.items()
        },
        "overall_violation_rate": sum(rule_violation_counts.values()) / total_responses_judged
        if total_responses_judged else 0.0,
    }
    with open(args.baseline_output, "w") as f:
        json.dump(baseline, f, indent=2)

    print(f"\nWrote {pairs_written} preference pairs to {out_path}")
    print(f"Wrote baseline violation stats to {args.baseline_output}")
    print(json.dumps(baseline, indent=2))


if __name__ == "__main__":
    main()
