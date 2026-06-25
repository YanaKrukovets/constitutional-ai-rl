"""
Fine-tunes the policy model on the preference pairs using LoRA + DPO
(Direct Preference Optimization).

Why DPO instead of PPO: PPO needs a separate reward model, a value model, and
rollout/sampling infrastructure during training, which is heavy and flaky on a
free Colab GPU. DPO only needs (prompt, chosen, rejected) triples and trains
directly against a closed-form objective derived from the same RLHF objective
PPO optimizes — it's also the direction the field has largely moved toward in
practice. We still call this RLAIF because the preference labels come from an
AI judge applying a written constitution (see judge.py), not human labelers.

Only the LoRA adapter is saved (a few MB), not the full model, so it's cheap to
store/share and easy to reload on top of the same base model later.
"""

import argparse
from pathlib import Path

import torch

# Some torch/transformers version combos (seen on macOS + torch 2.2.x) probe for
# torch.backends.mps.is_macos_or_newer, which doesn't exist in this torch build
# even though MPS itself works fine here. Shim it rather than pin older deps.
if not hasattr(torch.backends.mps, "is_macos_or_newer"):
    torch.backends.mps.is_macos_or_newer = lambda *args, **kwargs: True

from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preference-data", default=str(DATA_DIR / "preference_pairs.jsonl"))
    parser.add_argument("--base-model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR / "dpo_adapter"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--beta", type=float, default=0.1, help="DPO temperature; controls how strongly to penalize the rejected response")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--force-cpu", action="store_true", help="force CPU training (workaround for a PyTorch MPS kernel bug on some macOS/torch combos)")
    parser.add_argument("--max-steps", type=int, default=-1, help="override epochs with a fixed step count, for quick smoke tests")
    args = parser.parse_args()

    dataset = load_dataset("json", data_files=args.preference_data, split="train")
    # DPOTrainer expects exactly these column names
    dataset = dataset.remove_columns(
        [c for c in dataset.column_names if c not in ("prompt", "chosen", "rejected")]
    )
    print(f"Loaded {len(dataset)} preference pairs from {args.preference_data}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(args.base_model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    training_args = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        beta=args.beta,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        use_cpu=args.force_cpu,
        # model is small (0.5-1.5B) so checkpointing isn't needed; also avoids a
        # "gradients will be None" trap with PEFT models unless
        # model.enable_input_require_grads() is called first.
        gradient_checkpointing=False,
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    trainer.train()

    trainer.save_model(args.output_dir)
    print(f"Saved LoRA adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
