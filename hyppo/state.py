import json
from datetime import datetime, timezone
from pathlib import Path

from hyppo.config import (
    ensure_project_layout,
    hyppo_dir,
    logs_dir,
    project_config_path,
    skills_dir,
    state_dir,
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class WorkspaceState:
    def __init__(self, project_dir: str):
        self.project_dir = Path(project_dir).resolve()
        self.hyppo_dir = hyppo_dir(self.project_dir)
        self.state_dir = state_dir(self.project_dir)
        self.skills_dir = skills_dir(self.project_dir)
        self.logs_dir = logs_dir(self.project_dir)
        self.config_path = project_config_path(self.project_dir)

        self._active_runs: list[dict] | None = None
        self._completed_runs: list[dict] | None = None
        self._config: dict | None = None

    @classmethod
    def load_or_create(cls, project_dir: str) -> "WorkspaceState":
        state = cls(project_dir)
        ensure_project_layout(project_dir)

        if not state.config_path.exists():
            raise FileNotFoundError(f"Missing config file: {state.config_path}")
        if not (state.state_dir / "active_runs.json").exists():
            state._write_json("active_runs.json", [])
        if not (state.state_dir / "completed_runs.json").exists():
            state._write_json("completed_runs.json", [])

        return state

    def _read_json(self, filename: str) -> dict | list:
        return json.loads((self.state_dir / filename).read_text(encoding="utf-8"))

    def _write_json(self, filename: str, data: dict | list) -> None:
        (self.state_dir / filename).write_text(
            json.dumps(data, indent=2) + "\n",
            encoding="utf-8",
        )

    @property
    def config(self) -> dict:
        if self._config is None:
            self._config = json.loads(self.config_path.read_text(encoding="utf-8"))
        return self._config

    def search_space_exists(self) -> bool:
        return (self.state_dir / "search_space.json").exists()

    @property
    def search_space(self) -> dict | None:
        if not self.search_space_exists():
            return None
        return self._read_json("search_space.json")

    def read_search_space(self) -> dict | None:
        return self.search_space

    def write_search_space(self, data: dict) -> None:
        self._write_json("search_space.json", data)

    @property
    def active_runs(self) -> list[dict]:
        if self._active_runs is None:
            self._active_runs = self._read_json("active_runs.json")
        return self._active_runs

    def save_active_runs(self) -> None:
        self._write_json("active_runs.json", self.active_runs)

    def replace_active_runs(self, runs: list[dict]) -> None:
        self._active_runs = runs

    @property
    def completed_runs(self) -> list[dict]:
        if self._completed_runs is None:
            self._completed_runs = self._read_json("completed_runs.json")
        return self._completed_runs

    def save_completed_runs(self) -> None:
        self._write_json("completed_runs.json", self.completed_runs)

    @property
    def strategy(self) -> str:
        path = self.state_dir / "strategy.md"
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    def write_strategy(self, content: str) -> None:
        (self.state_dir / "strategy.md").write_text(content, encoding="utf-8")

    def next_run_number(self) -> int:
        all_runs = self.active_runs + self.completed_runs
        if not all_runs:
            return 1
        numbers = []
        for run in all_runs:
            run_id = run.get("run_id", "")
            try:
                numbers.append(int(run_id.split("_")[1]))
            except (IndexError, ValueError):
                continue
        return max(numbers, default=0) + 1

    def find_active_run(self, run_id: str) -> dict | None:
        return next((run for run in self.active_runs if run.get("run_id") == run_id), None)

    def wandb_run_path(self, run_id: str) -> str:
        entity = self.config.get("wandb_entity")
        project = self.config["wandb_project"]
        if entity:
            return f"{entity}/{project}/{run_id}"
        return f"{project}/{run_id}"

    def best_completed_val_loss(self) -> float | None:
        losses = [
            run["best_val_loss"]
            for run in self.completed_runs
            if isinstance(run.get("best_val_loss"), (int, float))
        ]
        if not losses:
            return None
        return min(losses)

    def save(self) -> None:
        if self._active_runs is not None:
            self.save_active_runs()
        if self._completed_runs is not None:
            self.save_completed_runs()

    def status_snapshot(self) -> dict:
        return {
            "active_runs": len(self.active_runs),
            "completed_runs": len(self.completed_runs),
            "best_val_loss": self.best_completed_val_loss(),
            "search_space_version": (
                self.search_space.get("version") if self.search_space_exists() else None
            ),
        }
