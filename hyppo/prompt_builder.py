import json
from pathlib import Path

from hyppo.state import WorkspaceState


def load_all_skills(skills_dir: Path) -> str:
    parts = []
    if not skills_dir.is_dir():
        return ""

    for path in sorted(skills_dir.glob("*.md")):
        parts.append(path.read_text().strip())
    return "\n\n---\n\n".join(part for part in parts if part)


def _format_metric(value) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if value is None:
        return "-"
    return str(value)


def _format_params(params: dict) -> str:
    if not params:
        return "-"
    return ", ".join(f"{key}={value}" for key, value in sorted(params.items()))


def _format_runs_table(runs: list[dict], title: str) -> str:
    if not runs:
        return f"## {title}\nNo runs."

    lines = [
        f"## {title}",
        "| run_id | status | epochs | best_val_loss | best_epoch | train_loss | trend | params |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for run in runs:
        lines.append(
            "| {run_id} | {status} | {epochs} | {best_val_loss} | {best_epoch} | "
            "{current_train_loss} | {trend} | {params} |".format(
                run_id=run.get("run_id", "-"),
                status=run.get("status", "running"),
                epochs=_format_metric(run.get("epochs_completed")),
                best_val_loss=_format_metric(run.get("best_val_loss")),
                best_epoch=_format_metric(run.get("best_epoch")),
                current_train_loss=_format_metric(run.get("current_train_loss")),
                trend=run.get("trend", "-"),
                params=_format_params(run.get("params", {})),
            )
        )
        last_losses = run.get("last_3_val_losses") or []
        if last_losses:
            lines.append(
                f"| ↳ recent val_loss | - | - | - | - | - | - | "
                f"{', '.join(_format_metric(loss) for loss in last_losses)} |"
            )
    return "\n".join(lines)


def format_state_for_prompt(state: WorkspaceState) -> str:
    sections = [
        "## Configuration\n```json\n" + json.dumps(state.config, indent=2) + "\n```",
    ]

    if state.search_space_exists():
        sections.append(
            "## Current Search Space\n```json\n"
            + json.dumps(state.search_space, indent=2)
            + "\n```"
        )
    else:
        sections.append("## Search Space\nNo search space defined yet.")

    sections.append(_format_runs_table(state.active_runs, "Active Runs"))

    if state.completed_runs:
        max_recent_runs = 10
        recent = state.completed_runs[-max_recent_runs:]
        older = state.completed_runs[:-max_recent_runs]
        if older:
            summary = [
                f"- {run.get('run_id', '?')}: best_val_loss={_format_metric(run.get('best_val_loss'))}"
                for run in older
            ]
            sections.append(
                "## Older Completed Runs\n"
                f"Summarized to keep prompt size bounded ({len(older)} runs).\n"
                + "\n".join(summary)
            )
        sections.append(_format_runs_table(recent, "Recent Completed Runs"))
    else:
        sections.append("## Completed Runs\nNo completed runs yet.")

    if state.strategy:
        sections.append("## Strategy\n" + state.strategy)

    return "\n\n".join(sections)


def build_prompt(state: WorkspaceState) -> str:
    skills_text = load_all_skills(state.skills_dir)
    state_text = format_state_for_prompt(state)

    prompt_parts = []
    if skills_text:
        prompt_parts.append(skills_text)
    prompt_parts.append("# Current State\n\n" + state_text)

    if not state.search_space_exists():
        prompt_parts.append(
            "## First Heartbeat Instructions\n"
            "No search space has been defined yet. Read the model description and available "
            "hyperparameters, then call `initialize_search_space` before launching any runs."
        )

    return "\n\n---\n\n".join(prompt_parts)
