import json
import os
import shutil
from pathlib import Path

CREDENTIALS_DIR = Path.home() / ".hyppo"
CREDENTIALS_FILE = CREDENTIALS_DIR / "credentials.json"

PACKAGE_SKILLS_DIR = Path(__file__).resolve().parent / "skills"


def hyppo_dir(project_dir: str | Path) -> Path:
    return Path(project_dir) / ".hyppo"


def state_dir(project_dir: str | Path) -> Path:
    return hyppo_dir(project_dir) / "state"


def skills_dir(project_dir: str | Path) -> Path:
    return hyppo_dir(project_dir) / "skills"


def logs_dir(project_dir: str | Path) -> Path:
    return hyppo_dir(project_dir) / "logs"


def legacy_project_config_path(project_dir: str | Path) -> Path:
    return Path(project_dir) / "hyppo.json"


def project_config_path(project_dir: str | Path) -> Path:
    return hyppo_dir(project_dir) / "hyppo.json"


def existing_project_config_path(project_dir: str | Path) -> Path:
    new_path = project_config_path(project_dir)
    if new_path.exists():
        return new_path

    legacy_path = legacy_project_config_path(project_dir)
    if legacy_path.exists():
        return legacy_path

    return new_path


def is_project_dir_writable(project_dir: str | Path) -> bool:
    project_path = Path(project_dir)
    if not project_path.is_dir():
        return False

    config_path = existing_project_config_path(project_path)
    if config_path.exists():
        return os.access(config_path, os.W_OK)

    return os.access(project_path, os.W_OK)


def ensure_project_layout(project_dir: str | Path) -> None:
    project_dir = Path(project_dir)
    hyppo_dir(project_dir).mkdir(parents=True, exist_ok=True)
    for d in [state_dir(project_dir), skills_dir(project_dir), logs_dir(project_dir)]:
        d.mkdir(parents=True, exist_ok=True)

    if PACKAGE_SKILLS_DIR.is_dir():
        for path in PACKAGE_SKILLS_DIR.glob("*.md"):
            target = skills_dir(project_dir) / path.name
            if not target.exists():
                shutil.copy2(path, target)


def load_project_config(project_dir: str | Path) -> dict:
    path = existing_project_config_path(project_dir)
    if not path.exists():
        raise FileNotFoundError(f"No hyppo config found in {project_dir}")
    return json.loads(path.read_text(encoding="utf-8"))


def save_project_config(project_dir: str | Path, config: dict) -> None:
    ensure_project_layout(project_dir)
    path = project_config_path(project_dir)
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def _load_credentials() -> dict:
    if not CREDENTIALS_FILE.exists():
        return {}
    return json.loads(CREDENTIALS_FILE.read_text(encoding="utf-8"))


def get_api_key(provider: str) -> str | None:
    env_vars = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }
    env_var = env_vars.get(provider)
    if env_var:
        val = os.environ.get(env_var)
        if val:
            return val

    return _load_credentials().get(provider)


def save_api_key(provider: str, key: str) -> None:
    CREDENTIALS_DIR.mkdir(parents=True, exist_ok=True)
    creds = _load_credentials()
    creds[provider] = key
    CREDENTIALS_FILE.write_text(
        json.dumps(creds, indent=2) + "\n",
        encoding="utf-8",
    )
    try:
        CREDENTIALS_FILE.chmod(0o600)
    except OSError:
        pass


class HyppoConfig:
    """High-level config object used by the CLI."""

    def __init__(self):
        self.project_dir: str | None = None
        self.script: str | None = None
        self.objective: str = "minimize"
        self.metric: str = "val_loss"
        self.llm_description: str = ""
        self.user_description: str = ""
        self.params: list[str] = []
        self.provider: str = "anthropic"
        self.model: str = "claude-sonnet-4-20250514"
        self.wandb_project: str = "hpo-agent"
        self.wandb_entity: str | None = None
        self.heartbeat_minutes: int = 5
        self.max_total_runs: int = 100
        self.max_concurrent_runs: int = 4
        self.max_time: int = 30
        self.modal_app: str = "hpo-agent"
        self.modal_function: str = "train_model"

    @property
    def description(self) -> str:
        parts = []
        if self.llm_description:
            parts.append(f"<llm_description>\n{self.llm_description}\n</llm_description>")
        if self.user_description:
            parts.append(f"<user_description>\n{self.user_description}\n</user_description>")
        return "\n\n".join(parts)

    def to_dict(self) -> dict:
        return {
            "objective": self.objective,
            "metric": self.metric,
            "project_path": self.project_dir or "",
            "training_script": self.script or "",
            "llm_description": self.llm_description,
            "user_description": self.user_description,
            "available_hyperparameters": self.params,
            "max_total_runs": self.max_total_runs,
            "max_concurrent_runs": self.max_concurrent_runs,
            "max_time": self.max_time,
            "heartbeat_interval_minutes": self.heartbeat_minutes,
            "wandb_entity": self.wandb_entity,
            "wandb_project": self.wandb_project,
            "llm_provider": self.provider,
            "llm_model": self.model,
            "modal_app_name": self.modal_app,
            "modal_function_name": self.modal_function,
        }

    def save(self) -> None:
        if not self.project_dir:
            return
        save_project_config(self.project_dir, self.to_dict())

    @classmethod
    def from_project(cls, project_dir: str) -> "HyppoConfig":
        data = load_project_config(project_dir)
        cfg = cls()
        cfg.project_dir = str(Path(project_dir).resolve())
        cfg.objective = data.get("objective", "minimize")
        cfg.metric = data.get("metric", "val_loss")
        cfg.llm_description = data.get("llm_description", data.get("model_description", ""))
        cfg.user_description = data.get("user_description", "")
        cfg.script = data.get("training_script", "") or None
        cfg.params = data.get("available_hyperparameters", [])
        cfg.provider = data.get("llm_provider", "anthropic")
        cfg.model = data.get("llm_model", "claude-sonnet-4-20250514")
        cfg.wandb_project = data.get("wandb_project", "hpo-agent")
        cfg.wandb_entity = data.get("wandb_entity")
        cfg.heartbeat_minutes = data.get("heartbeat_interval_minutes", 5)
        cfg.max_total_runs = data.get("max_total_runs", data.get("max_runs", 100))
        cfg.max_concurrent_runs = data.get("max_concurrent_runs", 4)
        cfg.max_time = data.get("max_time", data.get("max_epochs_per_run", 30))
        cfg.modal_app = data.get("modal_app_name", "hpo-agent")
        cfg.modal_function = data.get("modal_function_name", "train_model")
        return cfg

    def detect_script(self) -> str | None:
        if not self.project_dir:
            return None

        project_root = Path(self.project_dir)
        for name in ["train.py", "training.py", "main.py", "modal_app.py", "app.py"]:
            if (project_root / name).exists():
                return name

        training_dir = project_root / "training"
        if training_dir.is_dir():
            for f in sorted(training_dir.iterdir()):
                if f.suffix == ".py" and f.name != "__init__.py":
                    return f"training/{f.name}"

        examples_dir = project_root / "examples"
        if examples_dir.is_dir():
            for f in sorted(examples_dir.rglob("*.py")):
                if f.name != "__init__.py":
                    return f.relative_to(project_root).as_posix()

        return None

    def validate(self) -> list[str]:
        errors = []
        if not self.project_dir:
            errors.append("No project directory set. Use /project <path>")
        elif not is_project_dir_writable(self.project_dir):
            errors.append(
                f"Project directory is not writable: {self.project_dir}. "
                "Choose a directory where Hyppo can create .hyppo/"
            )

        if not self.script:
            errors.append("No training script set. Use /script <path> inside the project.")
        elif self.project_dir:
            script_path = (Path(self.project_dir) / self.script).resolve()
            project_root = Path(self.project_dir).resolve()
            try:
                script_path.relative_to(project_root)
            except ValueError:
                errors.append("Training script must live inside the project directory.")
            else:
                if not script_path.exists():
                    errors.append(f"Training script not found: {self.script}")

        if not self.description:
            errors.append(
                "No project description is available. Run /optimize after configuring the "
                "LLM or add notes with /describe."
            )

        if self.objective not in {"minimize", "maximize"}:
            errors.append("Objective must be either 'minimize' or 'maximize'")
        if not self.metric:
            errors.append("Metric must be a non-empty string")

        if self.heartbeat_minutes <= 0:
            errors.append("Heartbeat interval must be a positive integer")
        if self.max_total_runs <= 0:
            errors.append("Max total runs must be a positive integer")
        if self.max_concurrent_runs <= 0:
            errors.append("Max concurrent runs must be a positive integer")
        if self.max_concurrent_runs > self.max_total_runs:
            errors.append("Max concurrent runs cannot exceed max total runs")
        if self.max_time <= 0:
            errors.append("Max time must be a positive integer")
        if not self.modal_app:
            errors.append("No Modal app configured. Use /modal <app> <function>")
        if not self.modal_function:
            errors.append("No Modal function configured. Use /modal <app> <function>")

        api_key = get_api_key(self.provider)
        if not api_key:
            errors.append(
                f"No API key for {self.provider}. "
                "Use /apikey <key> or set the environment variable"
            )

        return errors
