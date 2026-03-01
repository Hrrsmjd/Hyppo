import json
import os
from datetime import datetime, timezone
from pathlib import Path


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class WorkspaceState:
    def __init__(self, workspace_dir: str):
        self.workspace_dir = Path(workspace_dir)
        self.state_dir = self.workspace_dir / "state"
        self.skills_dir = self.workspace_dir / "skills"
        self.plots_dir = self.workspace_dir / "plots"

        self._active_runs: list[dict] | None = None
        self._completed_runs: list[dict] | None = None
        self._config: dict | None = None

    @classmethod
    def load_or_create(cls, workspace_dir: str) -> "WorkspaceState":
        state = cls(workspace_dir)
        state.state_dir.mkdir(parents=True, exist_ok=True)
        state.plots_dir.mkdir(parents=True, exist_ok=True)

        # Initialize empty lists if files don't exist
        if not (state.state_dir / "active_runs.json").exists():
            state._write_json("active_runs.json", [])
        if not (state.state_dir / "completed_runs.json").exists():
            state._write_json("completed_runs.json", [])

        return state

    # --- JSON helpers ---

    def _read_json(self, filename: str) -> dict | list:
        path = self.state_dir / filename
        with open(path) as f:
            return json.load(f)

    def _write_json(self, filename: str, data: dict | list) -> None:
        path = self.state_dir / filename
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    # --- Config ---

    @property
    def config(self) -> dict:
        if self._config is None:
            self._config = self._read_json("config.json")
        return self._config

    # --- Search space ---

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

    # --- Active runs ---

    @property
    def active_runs(self) -> list[dict]:
        if self._active_runs is None:
            self._active_runs = self._read_json("active_runs.json")
        return self._active_runs

    def save_active_runs(self) -> None:
        self._write_json("active_runs.json", self.active_runs)

    # --- Completed runs ---

    @property
    def completed_runs(self) -> list[dict]:
        if self._completed_runs is None:
            self._completed_runs = self._read_json("completed_runs.json")
        return self._completed_runs

    def save_completed_runs(self) -> None:
        self._write_json("completed_runs.json", self.completed_runs)

    # --- Strategy ---

    @property
    def strategy(self) -> str:
        path = self.state_dir / "strategy.md"
        if not path.exists():
            return ""
        return path.read_text()

    def write_strategy(self, content: str) -> None:
        path = self.state_dir / "strategy.md"
        path.write_text(content)

    # --- Run helpers ---

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
        for run in self.active_runs:
            if run["run_id"] == run_id:
                return run
        return None

    def wandb_run_path(self, run_id: str) -> str:
        """Build the W&B API path for a run: 'entity/project/run_id' or 'project/run_id'."""
        entity = self.config.get("wandb_entity")
        project = self.config["wandb_project"]
        if entity:
            return f"{entity}/{project}/{run_id}"
        return f"{project}/{run_id}"

    def best_completed_val_loss(self) -> float | None:
        if not self.completed_runs:
            return None
        losses = [
            r["best_val_loss"]
            for r in self.completed_runs
            if isinstance(r.get("best_val_loss"), (int, float))
        ]
        if not losses:
            return None
        return min(losses)

    # --- Save all ---

    def save(self) -> None:
        if self._active_runs is not None:
            self.save_active_runs()
        if self._completed_runs is not None:
            self.save_completed_runs()
