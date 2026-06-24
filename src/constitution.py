"""
Parses constitution.md into a list of structured rules so other scripts
(judge.py, demo.py) don't have to hardcode rule text.

Each rule in constitution.md looks like:

## Rule 1 — No specific financial advice
Do not tell the user to buy/sell a specific stock...
- Why: ...
- Violation example: ...

This module turns that into:
  {"id": "rule_1", "title": "No specific financial advice",
   "description": "Do not tell the user to buy/sell...",
   "why": "...", "violation_example": "..."}
"""

from __future__ import annotations

import re
from pathlib import Path

CONSTITUTION_PATH = Path(__file__).resolve().parent.parent / "constitution.md"

RULE_HEADER_RE = re.compile(r"^##\s*Rule\s*(\d+)\s*—\s*(.+)$")


def load_rules(path: str | Path = CONSTITUTION_PATH) -> list[dict]:
    text = Path(path).read_text()
    lines = text.splitlines()

    rules = []
    current = None
    description_lines = []
    active_field = "description"

    def flush():
        if current is not None:
            current["description"] = " ".join(description_lines).strip()
            rules.append(current)

    for line in lines:
        header_match = RULE_HEADER_RE.match(line.strip())
        if header_match:
            flush()
            rule_num, title = header_match.groups()
            current = {
                "id": f"rule_{rule_num}",
                "title": title.strip(),
                "why": "",
                "violation_example": "",
            }
            description_lines = []
            active_field = "description"
            continue

        if current is None:
            continue

        stripped = line.strip()
        if stripped.startswith("- Why:"):
            current["why"] = stripped[len("- Why:"):].strip()
            active_field = "why"
        elif stripped.startswith("- Violation example:"):
            current["violation_example"] = stripped[len("- Violation example:"):].strip()
            active_field = "violation_example"
        elif stripped and not stripped.startswith("#"):
            if active_field == "description":
                description_lines.append(stripped)
            else:
                current[active_field] = (current[active_field] + " " + stripped).strip()

    flush()
    return rules


def format_for_prompt(rules: list[dict] | None = None) -> str:
    """Renders the rules as a numbered block to drop into the judge prompt."""
    if rules is None:
        rules = load_rules()
    blocks = []
    for r in rules:
        blocks.append(
            f"[{r['id']}] {r['title']}\n"
            f"  Rule: {r['description']}\n"
            f"  Why it matters: {r['why']}\n"
            f"  Example violation: {r['violation_example']}"
        )
    return "\n\n".join(blocks)


if __name__ == "__main__":
    rules = load_rules()
    print(f"Loaded {len(rules)} rules:\n")
    print(format_for_prompt(rules))
