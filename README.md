# Constitutional AI / RLAIF

A small, fully runnable implementation of Anthropic's Constitutional AI idea: write down
a set of rules, have an AI judge apply them to generate preference data (this is
**RLAIF — RL from AI Feedback**), fine-tune a model on that preference data with
**DPO**, then red-team the result and watch it get more robust across rounds.

Everything here runs on a free Colab T4 GPU with a small open model (Qwen2.5-0.5B-Instruct)
+ LoRA — see [`notebooks/colab_quickstart.ipynb`](notebooks/colab_quickstart.ipynb) to run
the whole thing end to end.

## Pipeline

```
constitution.md                 (6 written rules, each with a rationale + example violation)
       │
       ▼
generate_prompts.py              56 hand-written prompts: benign / boundary / adversarial
       │
       ▼
build_preference_dataset.py      sample 2 responses per prompt (temp 0.9) ──► judge.py
       │                         judge picks the better response per the constitution,
       │                         tags which rules each response violates
       ▼
data/preference_pairs.jsonl      {prompt, chosen, rejected}  ← this *is* the RLAIF step
       │
       ▼
train_dpo.py                     LoRA + DPO fine-tune on the preference pairs
       │
       ▼
outputs/dpo_adapter              a few-MB LoRA adapter (not a full model copy)
       │
       ├──► evaluate.py          base vs. tuned model on held-out prompts,
       │                         per-rule / per-bucket violation rates, win-rate,
       │                         response-length sanity check (over-refusal guard)
       │
       └──► red_team.py          attack the tuned model with adversarial prompts,
                                  retrain on failures (continuing the same adapter),
                                  repeat — watch jailbreak success rate drop by round

app/demo.py                      Gradio UI: base vs. tuned side by side, live judge
                                  breakdown, constitution toggle (default vs. relaxed)
```

## Why these design choices

**DPO instead of PPO.** PPO needs a separate reward model, a value model, and
rollout/sampling infrastructure during training — heavy and flaky on a free GPU.
DPO only needs `(prompt, chosen, rejected)` triples and trains directly against a
closed-form objective derived from the same RLHF objective PPO optimizes. It's also
the direction the field has largely moved in practice. This is still RLAIF, not
plain preference tuning, because the chosen/rejected labels come from an AI judge
applying a written constitution (`judge.py`), not from human labelers.

**LLM-as-judge instead of human labeling.** The whole point of Constitutional AI is
that a model can supervise itself against a written set of principles instead of
needing a human to label every example. `judge.py` supports two backends: the
Anthropic API (used automatically if `ANTHROPIC_API_KEY` is set) or a local
fallback model (Qwen2.5-1.5B-Instruct — deliberately bigger than the policy model
being trained), so the whole pipeline runs with zero API keys and zero cost.

**Small model + LoRA instead of a bigger model.** The point of this project is the
pipeline (constitution → judge → preference data → DPO → eval → red-team), not
raw model capability. A 0.5B model with a LoRA adapter makes every stage runnable
and inspectable on a free GPU in minutes, and the adapter is small enough to
actually share.

**Rejection sampling with a templated fallback in the red-team loop.** When the
model jailbreaks, `red_team.py` first tries to find a safe response by resampling
the *same* model a few times (temperature 0.9) and judging each resample — if the
model already "knows" a safe answer some fraction of the time, this distills that
behavior without needing a separate teacher model. If none of the resamples are
clean, it falls back to a templated refusal that cites the specific violated rule,
so every failure still produces a usable training pair.

## Known limitations

- **The local judge model is noisy.** Qwen2.5-1.5B-Instruct sometimes hallucinates
  rule violations on clearly benign responses (e.g. flagging a resume-writing tip
  as financial advice). This is a known, observed failure mode of using a small
  model as judge — swapping in `ANTHROPIC_API_KEY` fixes judge quality with zero
  code changes, since every script auto-detects it.
- **Small scale.** 56 hand-written prompts, a 0.5B policy model, and a handful of
  red-team rounds are enough to demonstrate that the mechanism works end to end,
  not to make a real claim about adversarial robustness or safety at production
  scale. Treat the red-teaming results as an illustrative analog of the technique,
  not a rigorous robustness evaluation.
- **Over-refusal isn't fully solved, just monitored.** `evaluate.py` tracks average
  response length and per-bucket violation rates specifically so a regression
  toward uselessly evasive behavior on benign prompts would show up — but nothing
  in the training loop actively optimizes against it beyond the judge's instruction
  to weigh helpfulness when rules are equally well followed.
- **This was developed and smoke-tested on a Mac (CPU/MPS), not the target GPU.**
  Two real local-only issues were hit and worked around: a missing
  `torch.backends.mps.is_macos_or_newer` API some transformers versions probe for
  (shimmed), and a PyTorch-on-Apple-Silicon Metal kernel bug that crashes DPO
  training on MPS entirely (worked around with `--force-cpu` for local testing).
  Neither should occur on Colab's CUDA GPU, which is the intended way to run the
  real training/eval/red-team steps — see the notebook.

## Running it

See [`notebooks/colab_quickstart.ipynb`](notebooks/colab_quickstart.ipynb) for the
full guided walkthrough. The short version, from this directory:

```bash
pip install -r requirements.txt

python src/generate_prompts.py
python src/build_preference_dataset.py      # the RLAIF step
python src/train_dpo.py                     # LoRA + DPO fine-tune
python src/evaluate.py                       # base vs. tuned, before/after metrics
python src/red_team.py --rounds 3            # adversarial robustness across rounds
python app/demo.py                           # interactive side-by-side demo
```

Results land in `results/metrics.json`, `results/red_team_metrics.json`, and
`results/plots/`.

## Project layout

```
constitution.md                  the written rules (default)
constitution_relaxed.md          a looser variant, used by the demo's toggle
requirements.txt
src/
  constitution.py                 parses constitution.md into structured rules
  generate_prompts.py             builds the benign/boundary/adversarial prompt set
  sample_responses.py             samples completions from the policy model
  judge.py                        AI judge: compare() for pairwise, classify() for single-response
  build_preference_dataset.py     orchestrates sampling + judging into DPO training pairs
  train_dpo.py                    LoRA + DPO fine-tuning (supports continuing from an existing adapter)
  evaluate.py                     base vs. tuned comparison on held-out prompts
  red_team.py                     adversarial robustness loop across retraining rounds
app/
  demo.py                         Gradio UI
notebooks/
  colab_quickstart.ipynb          end-to-end runnable notebook for a free T4 GPU
data/                             generated prompts, preference pairs, baseline stats
outputs/                          saved LoRA adapters (small, a few MB each)
results/                          metrics.json, red_team_metrics.json, plots/
```
