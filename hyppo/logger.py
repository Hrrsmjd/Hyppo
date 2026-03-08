import json
from datetime import datetime, timezone
from pathlib import Path


class MarkdownLogger:
    def __init__(self, logs_dir: Path):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.heartbeat_num = 0

    def new_heartbeat(self) -> None:
        self.heartbeat_num += 1

    def _timestamp(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def log_tool(self, tool_name: str, tool_input: dict, result: dict) -> None:
        path = self.logs_dir / "tool_log.md"
        entry = (
            f"\n### {tool_name}\n"
            f"_Heartbeat #{self.heartbeat_num} — {self._timestamp()}_\n\n"
            f"**Input:**\n```json\n{json.dumps(tool_input, indent=2)}\n```\n\n"
            f"**Result:**\n```json\n{json.dumps(result, indent=2)}\n```\n\n---\n"
        )
        with open(path, "a") as f:
            f.write(entry)

    def log_prompt(self, prompt: str) -> None:
        path = self.logs_dir / "llm_log.md"
        entry = (
            f"\n## Heartbeat #{self.heartbeat_num} — {self._timestamp()}\n\n"
            f"**Prompt length:** {len(prompt)} chars\n\n"
        )
        with open(path, "a") as f:
            f.write(entry)

    def log_response(self, text: str, finish_reason: str) -> None:
        path = self.logs_dir / "llm_log.md"
        entry = (
            f"**Finish reason:** {finish_reason}\n\n"
            f"### Response\n{text}\n\n---\n"
        )
        with open(path, "a") as f:
            f.write(entry)
