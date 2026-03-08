import json
from pathlib import Path

from hyppo.state import WorkspaceState


def load_all_skills(skills_dir: Path) -> str:
    parts = []
    if not skills_dir.is_dir():
        return ""

    for path in sorted(skills_dir.glob("*.md")):
        parts.append(path.read_text(encoding="utf-8").strip())
    return "\n\n---\n\n".join(part for part in parts if part)


def _format_metric(value, digits: int = 4) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    if value is None:
        return "-"
    return str(value)


def _format_params(params: dict) -> str:
    if not params:
        return "-"
    return ", ".join(f"{key}={value}" for key, value in sorted(params.items()))


def _format_history(history: list[dict]) -> str:
    if not history:
        return "No metric history yet."

    lines = [
        "| time_s | progress_% | val_loss | train_loss |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for point in history:
        lines.append(
            "| {time_seconds} | {progress_percent} | {val_loss} | {train_loss} |".format(
                time_seconds=_format_metric(point.get("time_seconds"), digits=1),
                progress_percent=_format_metric(point.get("progress_percent"), digits=1),
                val_loss=_format_metric(point.get("val_loss")),
                train_loss=_format_metric(point.get("train_loss")),
            )
        )
    return "\n".join(lines)


def _format_runs(runs: list[dict], title: str, include_history: bool) -> str:
    if not runs:
        return f"## {title}\nNo runs."

    lines = [
        f"## {title}",
        "| run_id | status | elapsed_s | progress_% | best_val_loss | best_time_s | trend | params |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for run in runs:
        lines.append(
            "| {run_id} | {status} | {elapsed_time_seconds} | {progress_percent} | "
            "{best_val_loss} | {best_time_seconds} | {trend} | {params} |".format(
                run_id=run.get("run_id", "-"),
                status=run.get("status", "running"),
                elapsed_time_seconds=_format_metric(run.get("elapsed_time_seconds"), digits=1),
                progress_percent=_format_metric(run.get("progress_percent"), digits=1),
                best_val_loss=_format_metric(run.get("best_val_loss")),
                best_time_seconds=_format_metric(run.get("best_time_seconds"), digits=1),
                trend=run.get("trend", "-"),
                params=_format_params(run.get("params", {})),
            )
        )
        if include_history:
            lines.append("")
            lines.append(f"### {run.get('run_id', '-') } Metric History")
            lines.append(_format_history(run.get("metric_history", [])))
            lines.append("")
    return "\n".join(lines).strip()


def format_state_for_prompt(state: WorkspaceState) -> str:
    config_for_prompt = dict(state.config)
    llm_description = config_for_prompt.pop("llm_description", "").strip()
    user_description = config_for_prompt.pop("user_description", "").strip()

    sections = [
        "## Configuration\n```json\n" + json.dumps(config_for_prompt, indent=2) + "\n```",
    ]

    description_parts = []
    if llm_description:
        description_parts.append(
            "<llm_description>\n" + llm_description + "\n</llm_description>"
        )
    if user_description:
        description_parts.append(
            "<user_description>\n" + user_description + "\n</user_description>"
        )
    if description_parts:
        sections.append("## Project Description\n" + "\n\n".join(description_parts))

    if state.search_space_exists():
        sections.append(
            "## Current Search Space\n```json\n"
            + json.dumps(state.search_space, indent=2)
            + "\n```"
        )
    else:
        sections.append("## Search Space\nNo search space defined yet.")

    run_limits = (
        f"Total runs started: {state.total_runs_started()} / {state.max_total_runs()}\n"
        f"Runs remaining: {state.runs_remaining()}\n"
        f"Active runs: {len(state.active_runs)} / {state.config.get('max_concurrent_runs', 4)}"
    )
    sections.append("## Run Limits\n" + run_limits)

    sections.append(_format_runs(state.active_runs, "Active Runs", include_history=True))

    if state.completed_runs:
        max_recent_runs = 10
        recent = state.completed_runs[-max_recent_runs:]
        older = state.completed_runs[:-max_recent_runs]
        if older:
            summary = [
                (
                    f"- {run.get('run_id', '?')}: best_val_loss={_format_metric(run.get('best_val_loss'))}, "
                    f"best_time_s={_format_metric(run.get('best_time_seconds'), digits=1)}"
                )
                for run in older
            ]
            sections.append(
                "## Older Completed Runs\n"
                f"Summarized to keep prompt size bounded ({len(older)} runs).\n"
                + "\n".join(summary)
            )
        sections.append(_format_runs(recent, "Recent Completed Runs", include_history=True))
    else:
        sections.append("## Completed Runs\nNo completed runs yet.")

    if state.strategy:
        sections.append("## Strategy\n" + state.strategy)

    if state.insights_history:
        sections.append("## Historical Insights\n" + state.insights_history)

    return "\n\n".join(sections)


def build_prompt(state: WorkspaceState) -> str:
    skills_text = load_all_skills(state.skills_dir)
    state_text = format_state_for_prompt(state)
    allowed_hyperparameters = state.config.get("available_hyperparameters", [])

    prompt_parts = []
    if skills_text:
        prompt_parts.append(skills_text)
    if allowed_hyperparameters:
        prompt_parts.append(
            "## Hyperparameter Guardrails\n"
            "Only use hyperparameters from `available_hyperparameters` when defining, "
            "updating, or launching from the search space.\n"
            f"Allowed hyperparameters: {', '.join(allowed_hyperparameters)}.\n"
            "Do not invent or add any other hyperparameters."
        )
    prompt_parts.append("# Current State\n\n" + state_text)

    if not state.search_space_exists():
        prompt_parts.append(
            "## First Heartbeat Instructions\n"
            "No search space has been defined yet. Read the project description and use only "
            "the allowed hyperparameters from `available_hyperparameters`, then call "
            "`initialize_search_space` before launching any runs."
        )

    if state.max_total_runs_reached():
        prompt_parts.append(
            "## Run Budget Constraint\n"
            "The max total run budget has been reached. Do not launch new runs. "
            "You may only update strategy or search-space notes."
        )

    return "\n\n---\n\n".join(prompt_parts)
