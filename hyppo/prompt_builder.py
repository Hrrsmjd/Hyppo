import json
from pathlib import Path

from hyppo.metrics import (
    get_metric_name,
    get_run_best_metric,
    get_run_best_time_seconds,
    get_run_latest_metric,
)
from hyppo.state import WorkspaceState

MAX_DESCRIPTION_CHARS = 3000
MAX_STRATEGY_CHARS = 2500
MAX_ACTIVE_HISTORY_POINTS = 5
MAX_RECENT_COMPLETED_RUNS = 6
MAX_OLDER_COMPLETED_SUMMARY = 6


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


def _truncate_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n\n[Truncated to keep prompt size bounded.]"


def _history_columns(history: list[dict], metric_name: str) -> list[str]:
    columns = ["time_seconds", "progress_percent", "metric"]
    if any(point.get("train_loss") is not None for point in history):
        columns.append("train_loss")
    if metric_name != "val_loss" and any(point.get("val_loss") is not None for point in history):
        columns.append("val_loss")
    if metric_name != "accuracy" and any(point.get("accuracy") is not None for point in history):
        columns.append("accuracy")
    return columns


def _column_label(column: str, metric_name: str) -> str:
    labels = {
        "time_seconds": "time_s",
        "progress_percent": "progress_%",
        "metric": metric_name,
        "train_loss": "train_loss",
        "val_loss": "val_loss",
        "accuracy": "accuracy",
    }
    return labels[column]


def _format_history(history: list[dict], metric_name: str, max_points: int | None = None) -> str:
    if not history:
        return "No metric history yet."

    displayed = history[-max_points:] if max_points else history
    columns = _history_columns(displayed, metric_name)
    header = "| " + " | ".join(_column_label(column, metric_name) for column in columns) + " |"
    divider = "| " + " | ".join("---:" for _ in columns) + " |"
    lines = [header, divider]
    for point in displayed:
        row = []
        for column in columns:
            digits = 1 if column in {"time_seconds", "progress_percent"} else 4
            row.append(_format_metric(point.get(column), digits=digits))
        lines.append("| " + " | ".join(row) + " |")
    if max_points and len(history) > len(displayed):
        lines.append(
            f"\nShowing the most recent {len(displayed)} of {len(history)} history points."
        )
    return "\n".join(lines)


def _format_runs(
    runs: list[dict],
    title: str,
    include_history: bool,
    config: dict,
    history_points: int | None = None,
) -> str:
    if not runs:
        return f"## {title}\nNo runs."

    metric_name = get_metric_name(config)
    lines = [
        f"## {title}",
        f"| run_id | status | elapsed_s | progress_% | best_{metric_name} | latest_{metric_name} | best_time_s | trend | params |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for run in runs:
        lines.append(
            "| {run_id} | {status} | {elapsed_time_seconds} | {progress_percent} | "
            "{best_metric} | {latest_metric} | {best_time_seconds} | {trend} | {params} |".format(
                run_id=run.get("run_id", "-"),
                status=run.get("status", "running"),
                elapsed_time_seconds=_format_metric(run.get("elapsed_time_seconds"), digits=1),
                progress_percent=_format_metric(run.get("progress_percent"), digits=1),
                best_metric=_format_metric(get_run_best_metric(run, config)),
                latest_metric=_format_metric(get_run_latest_metric(run, config)),
                best_time_seconds=_format_metric(get_run_best_time_seconds(run, config), digits=1),
                trend=run.get("trend", "-"),
                params=_format_params(run.get("params", {})),
            )
        )
        if include_history:
            lines.append("")
            lines.append(f"### {run.get('run_id', '-') } Metric History")
            lines.append(
                _format_history(
                    run.get("metric_history", []),
                    metric_name=metric_name,
                    max_points=history_points,
                )
            )
            lines.append("")
    return "\n".join(lines).strip()


def _config_for_prompt(config: dict) -> dict:
    return {
        "objective": config.get("objective", "minimize"),
        "metric": config.get("metric", "val_loss"),
        "training_script": config.get("training_script", ""),
        "max_total_runs": config.get("max_total_runs"),
        "max_concurrent_runs": config.get("max_concurrent_runs"),
        "max_time": config.get("max_time"),
    }


def _search_space_for_prompt(search_space: dict) -> dict:
    compact = {
        "version": search_space.get("version"),
        "parameters": search_space.get("parameters", {}),
    }
    changelog = search_space.get("changelog") or []
    if changelog:
        compact["latest_change"] = changelog[-1]
    return compact


def format_state_for_prompt(state: WorkspaceState) -> str:
    config_for_prompt = dict(state.config)
    llm_description = config_for_prompt.pop("llm_description", "").strip()
    user_description = config_for_prompt.pop("user_description", "").strip()

    sections = [
        "## Configuration\n```json\n" + json.dumps(_config_for_prompt(config_for_prompt), indent=2) + "\n```",
    ]

    description_parts = []
    if llm_description:
        description_parts.append(
            "<llm_description>\n"
            + _truncate_text(llm_description, MAX_DESCRIPTION_CHARS)
            + "\n</llm_description>"
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
            + json.dumps(_search_space_for_prompt(state.search_space), indent=2)
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

    sections.append(
        _format_runs(
            state.active_runs,
            "Active Runs",
            include_history=True,
            config=state.config,
            history_points=MAX_ACTIVE_HISTORY_POINTS,
        )
    )

    if state.completed_runs:
        recent = state.completed_runs[-MAX_RECENT_COMPLETED_RUNS:]
        older = state.completed_runs[:-MAX_RECENT_COMPLETED_RUNS]
        if older:
            summary = [
                (
                    f"- {run.get('run_id', '?')}: best_{get_metric_name(state.config)}="
                    f"{_format_metric(get_run_best_metric(run, state.config))}, "
                    f"best_time_s={_format_metric(get_run_best_time_seconds(run, state.config), digits=1)}"
                )
                for run in older[-MAX_OLDER_COMPLETED_SUMMARY:]
            ]
            sections.append(
                "## Older Completed Runs\n"
                f"Summarized to keep prompt size bounded ({len(older)} runs, showing the most recent {min(len(older), MAX_OLDER_COMPLETED_SUMMARY)}).\n"
                + "\n".join(summary)
            )
        sections.append(
            _format_runs(recent, "Recent Completed Runs", include_history=False, config=state.config)
        )
    else:
        sections.append("## Completed Runs\nNo completed runs yet.")

    if state.strategy:
        sections.append("## Strategy\n" + _truncate_text(state.strategy, MAX_STRATEGY_CHARS))

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

    prompt_parts.append(
        "## Heartbeat Sequencing Instructions\n"
        "Before launching any new runs on a heartbeat, first call `update_strategy` to record "
        "your observations, reasoning, and the plan for the runs you are about to launch. "
        "If you need to change the search space, do that before `launch_run` as well. "
        "Preferred order when launching is: `update_strategy`, optional `update_search_space`, "
        "then `launch_run`."
    )

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
