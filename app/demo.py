"""
Interactive demo: type a prompt, see the base model's response next to the
DPO-tuned model's response, with the judge's live rule-violation breakdown for
each. A constitution toggle lets you re-score both responses against a
different (looser) rule set, which demonstrates that the judge's verdicts
genuinely track the written rules rather than being hardcoded behavior.

Run with: python app/demo.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

if not hasattr(torch.backends.mps, "is_macos_or_newer"):
    torch.backends.mps.is_macos_or_newer = lambda *args, **kwargs: True

import gradio as gr
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from constitution import load_rules  # noqa: E402
from judge import Judge  # noqa: E402
from sample_responses import generate  # noqa: E402

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_DIR = ROOT_DIR / "outputs" / "dpo_adapter"
CONSTITUTIONS = {
    "Default (constitution.md)": ROOT_DIR / "constitution.md",
    "Relaxed (constitution_relaxed.md)": ROOT_DIR / "constitution_relaxed.md",
}

_state = {}


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_everything():
    if _state:
        return _state

    device = pick_device()
    print(f"Loading models on device: {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    base_model = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL)
    base_model.to(device)
    base_model.eval()

    if ADAPTER_DIR.exists():
        tuned_model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR))
        tuned_model.to(device)
        tuned_model.eval()
    else:
        print(f"No adapter found at {ADAPTER_DIR}, tuned model will mirror the base model.")
        tuned_model = base_model

    # JUDGE_DEVICE lets you force the judge model onto a different device than
    # the policy models, e.g. JUDGE_DEVICE=cpu, when they don't all fit on the
    # same small GPU memory pool (this is purely a local hardware constraint,
    # not something the deployed/Colab setup needs).
    judge = Judge(device=os.environ.get("JUDGE_DEVICE"))
    print(f"Judge backend: {judge.backend} ({judge.model})")

    _state.update(tokenizer=tokenizer, base_model=base_model, tuned_model=tuned_model,
                   judge=judge, device=device)
    return _state


def format_violations(violated_rules, rules_by_id):
    if not violated_rules:
        return "no rules violated"
    lines = []
    for rule_id in violated_rules:
        rule = rules_by_id.get(rule_id)
        title = rule["title"] if rule else rule_id
        lines.append(f"- **{rule_id}**: {title}")
    return "\n".join(lines)


def run(prompt: str, constitution_choice: str, max_new_tokens: int):
    if not prompt.strip():
        return "", "", "", ""

    state = load_everything()
    tokenizer, base_model, tuned_model, judge, device = (
        state["tokenizer"], state["base_model"], state["tuned_model"], state["judge"], state["device"]
    )

    constitution_path = CONSTITUTIONS[constitution_choice]
    rules = load_rules(constitution_path)
    rules_by_id = {r["id"]: r for r in rules}

    base_response = generate(tokenizer, base_model, device, prompt, max_new_tokens, temperature=0.0)
    tuned_response = generate(tokenizer, tuned_model, device, prompt, max_new_tokens, temperature=0.0)

    base_verdict = judge.classify(prompt, base_response, rules=rules)
    tuned_verdict = judge.classify(prompt, tuned_response, rules=rules)

    base_breakdown = format_violations(base_verdict["violated_rules"], rules_by_id) + f"\n\n_{base_verdict['rationale']}_"
    tuned_breakdown = format_violations(tuned_verdict["violated_rules"], rules_by_id) + f"\n\n_{tuned_verdict['rationale']}_"

    return base_response, base_breakdown, tuned_response, tuned_breakdown


def build_app():
    with gr.Blocks(title="Constitutional AI / RLAIF Demo") as demo:
        gr.Markdown(
            "# Constitutional AI / RLAIF Demo\n"
            "Type a prompt and compare the **base model** to the **DPO-tuned "
            "model** (trained on AI-judge preference pairs derived from a "
            "written constitution). Switch the constitution below to see the "
            "judge's verdicts change for the exact same responses."
        )
        with gr.Row():
            prompt_box = gr.Textbox(label="Prompt", placeholder="e.g. What stock should I buy right now?", lines=2)
        with gr.Row():
            constitution_dropdown = gr.Dropdown(
                choices=list(CONSTITUTIONS.keys()),
                value=list(CONSTITUTIONS.keys())[0],
                label="Constitution used for judging",
            )
            max_tokens_slider = gr.Slider(50, 300, value=150, step=10, label="Max new tokens")
        run_button = gr.Button("Generate & Score", variant="primary")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Base model")
                base_output = gr.Textbox(label="Response", lines=6)
                base_breakdown_output = gr.Markdown(label="Judge breakdown")
            with gr.Column():
                gr.Markdown("### DPO-tuned model")
                tuned_output = gr.Textbox(label="Response", lines=6)
                tuned_breakdown_output = gr.Markdown(label="Judge breakdown")

        run_button.click(
            fn=run,
            inputs=[prompt_box, constitution_dropdown, max_tokens_slider],
            outputs=[base_output, base_breakdown_output, tuned_output, tuned_breakdown_output],
        )
    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch()
