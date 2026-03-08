from pathlib import Path

from hyppo.llm_client import LLMClient

IGNORED_DIRS = {
    ".git",
    ".hyppo",
    ".idea",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "venv",
}

TEXT_EXTENSIONS = {
    ".cfg",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


def _iter_project_files(project_dir: Path) -> list[Path]:
    files = []
    for path in sorted(project_dir.rglob("*")):
        if not path.is_file():
            continue
        if any(part in IGNORED_DIRS for part in path.parts):
            continue
        if path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        files.append(path)
    return files


def build_project_context(
    project_dir: str | Path,
    script: str | None = None,
    max_chars: int = 120_000,
) -> str:
    root = Path(project_dir).resolve()
    selected = _iter_project_files(root)
    if not selected:
        return "No supported text files found in the project."

    if script:
        script_path = (root / script).resolve()
        if script_path in selected:
            selected.remove(script_path)
            selected.insert(0, script_path)

    sections = [
        f"Project root: {root}",
        f"Training script: {script or '(not set)'}",
    ]
    used_chars = sum(len(section) for section in sections)
    included = 0

    for path in selected:
        rel_path = path.relative_to(root)
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        section = f"\n\n## File: {rel_path}\n```text\n{content}\n```"
        if included > 0 and used_chars + len(section) > max_chars:
            remaining = len(selected) - included
            sections.append(
                f"\n\n[Truncated after {included} files to keep the request bounded. "
                f"{remaining} files were omitted.]"
            )
            break

        sections.append(section)
        used_chars += len(section)
        included += 1

    return "".join(sections)


def generate_project_description(
    project_dir: str | Path,
    script: str | None,
    provider: str,
    model: str,
    logger=None,
) -> str:
    context = build_project_context(project_dir, script)
    prompt = (
        "You are preparing an autonomous hyperparameter optimization campaign.\n"
        "Read the supplied project files and write a concise but information-dense project "
        "description for the optimizer.\n\n"
        "Focus on:\n"
        "- task/objective\n"
        "- model architecture or algorithm\n"
        "- data flow and preprocessing\n"
        "- training loop and stopping behavior\n"
        "- likely hyperparameters worth tuning\n"
        "- any constraints or gotchas the optimizer should know\n\n"
        "Return plain markdown. Do not invent details that are not supported by the files.\n\n"
        f"{context}"
    )

    client = LLMClient(provider, model)
    response = client.chat(messages=[{"role": "user", "content": prompt}])
    description = response.choices[0].message.content or ""

    if logger:
        logger.log_prompt(prompt, title="Project Description Prompt")
        logger.log_response(
            description,
            response.choices[0].finish_reason,
            title="Project Description Response",
        )

    return description.strip()
