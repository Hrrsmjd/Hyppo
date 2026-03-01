import base64
import json
import os

from state import WorkspaceState


def load_all_skills(skills_dir: str) -> str:
    """Read and concatenate all .md files from the skills directory."""
    skills_dir_path = skills_dir
    parts = []
    if not os.path.isdir(skills_dir_path):
        return ""
    for filename in sorted(os.listdir(skills_dir_path)):
        if filename.endswith(".md"):
            filepath = os.path.join(skills_dir_path, filename)
            with open(filepath) as f:
                parts.append(f.read())
    return "\n\n---\n\n".join(parts)


def format_state_for_prompt(state: WorkspaceState) -> str:
    """Format all state into readable text for the prompt."""
    sections = []

    # Config
    sections.append("## Configuration\n```json\n" + json.dumps(state.config, indent=2) + "\n```")

    # Search space
    if state.search_space_exists():
        sections.append(
            "## Current Search Space\n```json\n"
            + json.dumps(state.search_space, indent=2)
            + "\n```"
        )
    else:
        sections.append("## Search Space\nNo search space defined yet.")

    # Active runs
    if state.active_runs:
        sections.append(
            "## Active Runs\n```json\n"
            + json.dumps(state.active_runs, indent=2)
            + "\n```"
        )
    else:
        sections.append("## Active Runs\nNo active runs.")

    # Completed runs — include recent runs in full, summarize older ones
    MAX_RECENT_RUNS = 10
    if state.completed_runs:
        recent = state.completed_runs[-MAX_RECENT_RUNS:]
        older = state.completed_runs[:-MAX_RECENT_RUNS]
        parts = []
        if older:
            summary_lines = []
            for r in older:
                val = r.get("best_val_loss")
                val_str = f"{val:.4f}" if isinstance(val, (int, float)) else str(val)
                summary_lines.append(f"  {r.get('run_id', '?')}: best_val_loss={val_str}")
            parts.append(
                f"### Older Runs ({len(older)} runs, summarized)\n" + "\n".join(summary_lines)
            )
        parts.append(
            "### Recent Runs\n```json\n" + json.dumps(recent, indent=2) + "\n```"
        )
        sections.append("## Completed Runs\n" + "\n\n".join(parts))
    else:
        sections.append("## Completed Runs\nNo completed runs yet.")

    # Strategy
    strategy = state.strategy
    if strategy:
        sections.append("## Strategy\n" + strategy)

    return "\n\n".join(sections)


def build_prompt_content(
    state: WorkspaceState, plot_paths: list[str]
) -> list[dict]:
    """Build content blocks (text + base64 images) for the Claude API call."""
    skills_text = load_all_skills(str(state.skills_dir))
    state_text = format_state_for_prompt(state)

    prompt = f"{skills_text}\n\n---\n\n# Current State\n\n{state_text}"

    # First-heartbeat detection
    if not state.search_space_exists():
        prompt += (
            "\n\n---\n\n"
            "**IMPORTANT: No search space has been defined yet. This is your first heartbeat.**\n"
            "Read the model description and available hyperparameters in the configuration above, "
            "then use the `initialize_search_space` tool to define your initial search space "
            "before launching any runs."
        )

    content: list[dict] = [{"type": "text", "text": prompt}]

    for path in plot_paths:
        with open(path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": img_b64,
                },
            }
        )

    return content
