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


def project_config_path(project_dir: str | Path) -> Path:
    return Path(project_dir) / "hyppo.json"


def ensure_project_layout(project_dir: str | Path) -> None:
    project_dir = Path(project_dir)
    for d in [state_dir(project_dir), skills_dir(project_dir), logs_dir(project_dir)]:
        d.mkdir(parents=True, exist_ok=True)

    # Copy default skills if missing
    if PACKAGE_SKILLS_DIR.is_dir():
        for path in PACKAGE_SKILLS_DIR.glob("*.md"):
            target = skills_dir(project_dir) / path.name
            if not target.exists():
                shutil.copy2(path, target)


def load_project_config(project_dir: str | Path) -> dict:
    path = project_config_path(project_dir)
    if not path.exists():
        raise FileNotFoundError(f"No hyppo.json in {project_dir}")
    return json.loads(path.read_text(encoding="utf-8"))


def save_project_config(project_dir: str | Path, config: dict) -> None:
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
        self.description: str = ""
        self.params: list[str] = []
        self.provider: str = "anthropic"
        self.model: str = "claude-sonnet-4-20250514"
        self.wandb_project: str = "hpo-agent"
        self.wandb_entity: str | None = None
        self.heartbeat_minutes: int = 5
        self.max_runs: int = 4
        self.max_epochs: int = 30
        self.modal_app: str = "hpo-agent"
        self.modal_function: str = "train_model"

    def to_dict(self) -> dict:
        return {
            "objective": "minimize",
            "metric": "val_loss",
            "model_description": self.description,
            "training_script": self.script or "",
            "available_hyperparameters": self.params,
            "max_concurrent_runs": self.max_runs,
            "max_epochs_per_run": self.max_epochs,
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
        cfg.project_dir = project_dir
        cfg.description = data.get("model_description", "")
        cfg.script = data.get("training_script", "")
        cfg.params = data.get("available_hyperparameters", [])
        cfg.provider = data.get("llm_provider", "anthropic")
        cfg.model = data.get("llm_model", "claude-sonnet-4-20250514")
        cfg.wandb_project = data.get("wandb_project", "hpo-agent")
        cfg.wandb_entity = data.get("wandb_entity")
        cfg.heartbeat_minutes = data.get("heartbeat_interval_minutes", 5)
        cfg.max_runs = data.get("max_concurrent_runs", 4)
        cfg.max_epochs = data.get("max_epochs_per_run", 30)
        cfg.modal_app = data.get("modal_app_name", "hpo-agent")
        cfg.modal_function = data.get("modal_function_name", "train_model")
        return cfg

    def detect_script(self) -> str | None:
        if not self.project_dir:
            return None
        for name in ["train.py", "training.py", "main.py"]:
            if (Path(self.project_dir) / name).exists():
                return name
        training_dir = Path(self.project_dir) / "training"
        if training_dir.is_dir():
            for f in training_dir.iterdir():
                if f.suffix == ".py" and f.name != "__init__.py":
                    return f"training/{f.name}"
        return None

    def validate(self) -> list[str]:
        errors = []
        if not self.project_dir:
            errors.append("No project directory set. Use /project <path>")
        if not self.description:
            errors.append("No model description. Use /describe <text>")
        if self.heartbeat_minutes <= 0:
            errors.append("Heartbeat interval must be a positive integer")
        if self.max_runs <= 0:
            errors.append("Max concurrent runs must be a positive integer")
        if self.max_epochs <= 0:
            errors.append("Max epochs per run must be a positive integer")
        if not self.modal_app:
            errors.append("No Modal app configured. Use /modal <app> <function>")
        if not self.modal_function:
            errors.append("No Modal function configured. Use /modal <app> <function>")
        api_key = get_api_key(self.provider)
        if not api_key:
            errors.append(
                f"No API key for {self.provider}. "
                f"Use /apikey <key> or set the environment variable"
            )
        return errors
