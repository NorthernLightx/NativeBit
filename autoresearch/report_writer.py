"""Update docs/RESEARCH_REPORT.md with autoresearch trial results."""

import os
import re
from datetime import datetime, timezone


def update_research_report(trial, report_path: str = "docs/RESEARCH_REPORT.md"):
    """Append a trial result to the research report's trial table."""
    if not os.path.exists(report_path):
        return

    with open(report_path, "r") as f:
        content = f.read()

    # Build the trial row
    config_str = ", ".join(
        f"{k}={v}" for k, v in sorted(trial.config.items())
        if v != None and k in ("requantize_every", "ema_decay", "block_size",
                                "n_codebook", "delay_quant_steps", "learning_rate")
    )

    screen_ppl = f"{trial.screen_ppl:.1f}" if hasattr(trial, "screen_ppl") and trial.screen_ppl else "—"
    validate_ppl = f"{trial.validate_ppl:.1f}" if hasattr(trial, "validate_ppl") and trial.validate_ppl else "—"
    confirm_ppl = f"{trial.confirm_ppl:.1f} ± {trial.confirm_std:.1f}" if hasattr(trial, "confirm_ppl") and trial.confirm_ppl else "—"
    sps = f"{trial.config.get('_steps_per_sec', '—')}"

    row = f"| {trial.trial_id} | {config_str} | {screen_ppl} | {validate_ppl} | {confirm_ppl} | {trial.status} | {sps} |"

    # Replace the awaiting line or append after the table header
    if "*(awaiting trials)*" in content:
        content = content.replace("| *(awaiting trials)* | | | | | | |", row)
    else:
        # Find the trial table and append
        table_pattern = r"(\| Trial \| Config \|.*?\n\|[-| ]+\n)(.*?)(\n\n)"
        match = re.search(table_pattern, content, re.DOTALL)
        if match:
            existing_rows = match.group(2)
            new_rows = existing_rows.rstrip() + "\n" + row
            content = content[:match.start(2)] + new_rows + content[match.start(3):]

    # Update best configuration section if this trial is accepted
    if trial.status == "accepted" and hasattr(trial, "confirm_ppl") and trial.confirm_ppl:
        best_section = f"""*(Updated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')})*

**Trial {trial.trial_id}** — Confirm PPL: {trial.confirm_ppl:.1f} ± {trial.confirm_std:.1f}

```
{config_str}
```"""
        content = re.sub(
            r"\*\(To be determined by autoresearch\)\*",
            best_section,
            content,
        )
        # Also update if there's already a best config
        content = re.sub(
            r"\*\(Updated .*?\)\*\n\n\*\*Trial \d+\*\*.*?```\n.*?\n```",
            best_section,
            content,
            flags=re.DOTALL,
        )

    with open(report_path, "w") as f:
        f.write(content)


def add_finding(finding: str, report_path: str = "docs/RESEARCH_REPORT.md"):
    """Add a key finding to section 5.4."""
    if not os.path.exists(report_path):
        return

    with open(report_path, "r") as f:
        content = f.read()

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    entry = f"- [{timestamp}] {finding}"

    if "*(Updated after each significant trial)*" in content:
        content = content.replace(
            "*(Updated after each significant trial)*",
            entry,
        )
    else:
        # Append to existing findings
        pattern = r"(### 5\.4 Key Findings\n)(.*?)(\n## )"
        match = re.search(pattern, content, re.DOTALL)
        if match:
            existing = match.group(2).rstrip()
            content = content[:match.start(2)] + existing + "\n" + entry + "\n" + content[match.start(3):]

    with open(report_path, "w") as f:
        f.write(content)
