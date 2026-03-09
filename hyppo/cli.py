import os
import threading
from dataclasses import dataclass
from pathlib import Path

from hyppo import __version__
from hyppo.config import (
    HyppoConfig,
    ensure_project_layout,
    existing_project_config_path,
    get_api_key,
    is_project_dir_writable,
    logs_dir,
    save_api_key,
)
from hyppo.logger import MarkdownLogger
from hyppo.project_context import generate_project_description

COMMAND_SPECS = (
    ("Setup commands:", (
        ("/project", "<path>", "Set project directory and infer an LLM description"),
        ("/script", "<path>", "Set training script path inside the project"),
        ("/describe", "<text>", "Append extra user notes to the project description"),
        ("/params", "<list>", "Set hyperparameters (comma-separated)"),
        ("/provider", "<name>", "Set LLM provider (anthropic/openai/openrouter)"),
        ("/model", "<name>", "Set LLM model"),
        ("/apikey", "<key>", "Set API key for current provider"),
        ("/wandb", "<project>", "Set W&B project (optionally entity/project)"),
        ("/heartbeat", "<mins>", "Set heartbeat interval in minutes"),
        ("/max_total_runs", "<n>", "Set max total runs across the whole campaign"),
        ("/max_concurrent_runs", "<n>", "Set max concurrent runs"),
        ("/max_time", "<mins>", "Set max runtime per run in minutes"),
        ("/modal", "<app> <func>", "Set Modal app and function name"),
        ("/config", "", "Show current configuration"),
    )),
    ("Campaign commands:", (
        ("/optimize", "", "Start the optimization campaign"),
        ("/status", "", "Show current campaign status"),
        ("/stop", "", "Stop campaign (if running)"),
        ("/quit", "", "Exit Hyppo"),
    )),
)

COMMAND_HELP = {
    name: description
    for _, commands in COMMAND_SPECS
    for name, _, description in commands
}
PATH_ARGUMENT_COMMANDS = {"/project", "/script"}
DEFAULT_HISTORY_PATH = Path.home() / ".hyppo" / "history"


@dataclass(frozen=True)
class CompletionCandidate:
    text: str
    start_position: int
    display_meta: str = ""


def print_banner():
    print(f"Welcome to Hyppo {__version__} - autonomous hyperparameter optimization.")
    print("Type /help for available commands.\n")


def print_help():
    lines = [""]
    for header, commands in COMMAND_SPECS:
        lines.append(header)
        for name, args, description in commands:
            suffix = f" {args}" if args else ""
            lines.append(f"  {name}{suffix:<26} {description}")
        lines.append("")
    print("\n".join(lines))


def _preview(text: str, limit: int = 80) -> str:
    if not text:
        return "NOT SET"
    return text[:limit] + ("..." if len(text) > limit else "")


def print_config(cfg: HyppoConfig):
    api_key = get_api_key(cfg.provider)
    key_status = f"***{api_key[-4:]}" if api_key else "NOT SET"

    print(
        f"""
  Project:             {cfg.project_dir or 'NOT SET'}
  Script:              {cfg.script or 'NOT SET'}
  LLM Description:     {_preview(cfg.llm_description)}
  User Description:    {_preview(cfg.user_description)}
  Params:              {', '.join(cfg.params) if cfg.params else 'NOT SET'}
  Provider:            {cfg.provider}
  Model:               {cfg.model}
  API Key:             {key_status}
  W&B Project:         {cfg.wandb_project}
  W&B Entity:          {cfg.wandb_entity or '(default)'}
  Heartbeat:           {cfg.heartbeat_minutes}m
  Max Total Runs:      {cfg.max_total_runs}
  Max Concurrent Runs: {cfg.max_concurrent_runs}
  Max Time / Run:      {cfg.max_time}m
  Modal App:           {cfg.modal_app}
  Modal Func:          {cfg.modal_function}
"""
    )


def _parse_positive_int(arg: str, label: str) -> int | None:
    try:
        value = int(arg)
    except ValueError:
        print(f"Usage: {label}")
        return None
    if value <= 0:
        print(f"Value must be positive: {value}")
        return None
    return value


def print_status(cfg: HyppoConfig):
    if not cfg.project_dir:
        print("No project selected. Use /project <path> first.")
        return

    from hyppo.state import WorkspaceState

    try:
        state = WorkspaceState.load_or_create(cfg.project_dir)
    except FileNotFoundError:
        print("No campaign state found yet. Run /optimize to create it.")
        return

    snapshot = state.status_snapshot()
    best = snapshot["best_val_loss"]
    best_text = f"{best:.4f}" if isinstance(best, (int, float)) else "N/A"
    space_version = snapshot["search_space_version"] or "N/A"

    print(
        f"""
  Active Runs:           {snapshot["active_runs"]}
  Completed Runs:        {snapshot["completed_runs"]}
  Total Runs Started:    {snapshot["total_runs_started"]}
  Runs Remaining:        {snapshot["runs_remaining"]}
  Best val_loss:         {best_text}
  Search Space Version:  {space_version}
"""
    )


def _normalize_script_path(arg: str, cfg: HyppoConfig) -> str | None:
    if not cfg.project_dir:
        print("Set /project first so the script can be resolved inside it.")
        return None

    raw_path = Path(os.path.expanduser(arg))
    script_path = raw_path if raw_path.is_absolute() else Path(cfg.project_dir) / raw_path
    script_path = script_path.resolve()
    project_root = Path(cfg.project_dir).resolve()

    try:
        relative_path = script_path.relative_to(project_root)
    except ValueError:
        print("Training script must be inside the project directory.")
        return None

    if not script_path.is_file():
        print(f"Script not found: {script_path}")
        return None

    return relative_path.as_posix()


def _append_description(existing: str, new_text: str) -> str:
    new_text = new_text.strip().strip('"').strip("'")
    if not new_text:
        return existing
    if not existing:
        return new_text
    return existing.rstrip() + "\n\n" + new_text


def _maybe_generate_description(cfg: HyppoConfig, force: bool = False) -> bool:
    if not cfg.project_dir:
        return False
    if cfg.llm_description and not force:
        return False

    if not get_api_key(cfg.provider):
        print("Project description generation deferred until an API key is configured.")
        return False

    try:
        ensure_project_layout(cfg.project_dir)
        logger = MarkdownLogger(logs_dir(cfg.project_dir))
        description = generate_project_description(
            cfg.project_dir,
            cfg.script,
            cfg.provider,
            cfg.model,
            logger=logger,
        )
    except Exception as exc:
        print(f"Project description generation failed: {exc}")
        return False

    if not description:
        print("Project description generation returned an empty response.")
        return False

    cfg.llm_description = description
    cfg.save()
    print("LLM project description refreshed.")
    return True


def _split_completion_context(text_before_cursor: str) -> tuple[list[str], str]:
    if not text_before_cursor:
        return [], ""

    tokens = text_before_cursor.split()
    if text_before_cursor[-1].isspace():
        return tokens, ""
    if not tokens:
        return [], text_before_cursor
    return tokens[:-1], tokens[-1]


def _path_completion_base(command: str, cfg: HyppoConfig, cwd: str) -> Path:
    if command == "/script" and cfg.project_dir:
        return Path(cfg.project_dir)
    return Path(cwd)


def _split_path_prefix(prefix: str) -> tuple[str, str]:
    normalized = prefix.replace("\\", "/")
    if normalized.endswith("/"):
        return normalized.rstrip("/"), ""
    parent_text, sep, stem = normalized.rpartition("/")
    if not sep:
        return "", normalized
    if not parent_text and normalized.startswith("/"):
        return "/", stem
    return parent_text, stem


def _resolve_completion_dir(base_dir: Path, parent_text: str) -> Path:
    if parent_text == "/":
        return Path("/")
    if parent_text.startswith("~"):
        return Path(os.path.expanduser(parent_text))
    if parent_text.startswith("/"):
        return Path(parent_text)
    return (base_dir / parent_text).resolve() if parent_text else base_dir.resolve()


def _candidate_prefix(parent_text: str) -> str:
    if not parent_text:
        return ""
    if parent_text == "/":
        return "/"
    return parent_text.rstrip("/") + "/"


def _path_completion_candidates(base_dir: Path, prefix: str, directories_only: bool) -> list[str]:
    parent_text, stem = _split_path_prefix(prefix)
    search_dir = _resolve_completion_dir(base_dir, parent_text)
    display_prefix = _candidate_prefix(parent_text)

    if not search_dir.is_dir():
        return []

    show_hidden = stem.startswith(".")
    matches = []

    for entry in search_dir.iterdir():
        if not show_hidden and entry.name.startswith("."):
            continue
        if directories_only and not entry.is_dir():
            continue
        if stem and not entry.name.startswith(stem):
            continue
        candidate = f"{display_prefix}{entry.name}"
        if entry.is_dir():
            candidate += "/"
        matches.append((not entry.is_dir(), entry.name.lower(), candidate))

    return [candidate for _, _, candidate in sorted(matches)]


def get_completion_candidates(
    text_before_cursor: str,
    cfg: HyppoConfig,
    cwd: str | None = None,
) -> list[CompletionCandidate]:
    cwd = cwd or os.getcwd()
    prior_tokens, current_token = _split_completion_context(text_before_cursor)

    if not prior_tokens and current_token.startswith("/"):
        matches = [
            CompletionCandidate(
                text=name,
                start_position=-len(current_token),
                display_meta=COMMAND_HELP[name],
            )
            for name in sorted(COMMAND_HELP)
            if name.startswith(current_token)
        ]
        return matches

    if len(prior_tokens) != 1:
        return []

    command = prior_tokens[0].lower()
    if command not in PATH_ARGUMENT_COMMANDS or not current_token.startswith("@"):
        return []

    candidates = _path_completion_candidates(
        _path_completion_base(command, cfg, cwd),
        current_token[1:],
        directories_only=command == "/project",
    )
    return [
        CompletionCandidate(
            text=f"@{candidate}",
            start_position=-len(current_token),
            display_meta="path",
        )
        for candidate in candidates
    ]


def normalize_interactive_line(line: str, cfg: HyppoConfig, cwd: str | None = None) -> str:
    del cfg, cwd
    stripped = line.strip()
    if not stripped:
        return ""

    parts = stripped.split(maxsplit=1)
    command = parts[0].lower()
    if command not in PATH_ARGUMENT_COMMANDS or len(parts) == 1:
        return stripped

    argument = parts[1].strip()
    if not argument.startswith("@"):
        return stripped
    return f"{command} {argument[1:]}"


def handle_command(line: str, cfg: HyppoConfig, campaign_stop_event: threading.Event | None) -> str | None:
    parts = line.strip().split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""
    config_changed = False

    if cmd == "/help":
        print_help()

    elif cmd == "/project":
        if not arg:
            print("Usage: /project <path>")
            return None
        path = os.path.abspath(os.path.expanduser(arg))
        if not os.path.isdir(path):
            print(f"Directory not found: {path}")
            return None
        if not is_project_dir_writable(path):
            print(
                "Project directory is not writable: "
                f"{path}. Choose a directory where Hyppo can write .hyppo/."
            )
            return None

        cfg.project_dir = path
        cfg.script = cfg.detect_script()
        config_changed = True

        print(f"Project directory: {cfg.project_dir}")
        if cfg.script:
            print(f"Detected training script: {cfg.script}")
        else:
            print("No training script detected. Set one with /script <path>.")

        _maybe_generate_description(cfg, force=True)

    elif cmd == "/script":
        if not arg:
            print("Usage: /script <path>")
            return None
        script = _normalize_script_path(arg, cfg)
        if not script:
            return None
        cfg.script = script
        config_changed = True
        print(f"Training script: {cfg.script}")
        _maybe_generate_description(cfg, force=True)

    elif cmd == "/describe":
        if not arg:
            print("Usage: /describe <text>")
            return None
        cfg.user_description = _append_description(cfg.user_description, arg)
        config_changed = True
        print(f"User description updated ({len(cfg.user_description)} chars)")

    elif cmd == "/params":
        if not arg:
            print("Usage: /params learning_rate,dropout,batch_size")
            return None
        cfg.params = [p.strip() for p in arg.split(",") if p.strip()]
        config_changed = True
        print(f"Parameters: {', '.join(cfg.params)}")

    elif cmd == "/provider":
        if arg not in ("anthropic", "openai", "openrouter"):
            print("Provider must be: anthropic, openai, or openrouter")
            return None
        cfg.provider = arg
        config_changed = True
        print(f"Provider: {arg}")
        _maybe_generate_description(cfg, force=False)

    elif cmd == "/model":
        if not arg:
            print("Usage: /model <model-name>")
            return None
        cfg.model = arg
        config_changed = True
        print(f"Model: {arg}")
        _maybe_generate_description(cfg, force=False)

    elif cmd == "/apikey":
        if not arg:
            print("Usage: /apikey <key>")
            return None
        save_api_key(cfg.provider, arg)
        print(f"API key saved for {cfg.provider}")
        _maybe_generate_description(cfg, force=False)

    elif cmd == "/wandb":
        if not arg:
            print("Usage: /wandb <project> or /wandb <entity/project>")
            return None
        if "/" in arg:
            entity, project = arg.split("/", 1)
            cfg.wandb_entity = entity
            cfg.wandb_project = project
        else:
            cfg.wandb_project = arg
        config_changed = True
        print(f"W&B: {cfg.wandb_entity or '(default)'}/{cfg.wandb_project}")

    elif cmd == "/heartbeat":
        value = _parse_positive_int(arg, "/heartbeat <minutes>")
        if value is not None:
            cfg.heartbeat_minutes = value
            config_changed = True
            print(f"Heartbeat interval: {cfg.heartbeat_minutes}m")

    elif cmd == "/max_total_runs":
        value = _parse_positive_int(arg, "/max_total_runs <n>")
        if value is not None:
            cfg.max_total_runs = value
            config_changed = True
            print(f"Max total runs: {cfg.max_total_runs}")

    elif cmd == "/max_concurrent_runs":
        value = _parse_positive_int(arg, "/max_concurrent_runs <n>")
        if value is not None:
            cfg.max_concurrent_runs = value
            config_changed = True
            print(f"Max concurrent runs: {cfg.max_concurrent_runs}")

    elif cmd == "/max_time":
        value = _parse_positive_int(arg, "/max_time <minutes>")
        if value is not None:
            cfg.max_time = value
            config_changed = True
            print(f"Max time per run: {cfg.max_time}m")

    elif cmd == "/modal":
        parts_modal = arg.split()
        if len(parts_modal) == 2:
            cfg.modal_app, cfg.modal_function = parts_modal
            config_changed = True
            print(f"Modal: {cfg.modal_app}::{cfg.modal_function}")
        elif len(parts_modal) == 1:
            cfg.modal_app = parts_modal[0]
            config_changed = True
            print(f"Modal app: {cfg.modal_app} (function: {cfg.modal_function})")
        else:
            print("Usage: /modal <app_name> <function_name>")

    elif cmd == "/config":
        print_config(cfg)

    elif cmd == "/status":
        print_status(cfg)

    elif cmd == "/optimize":
        return "optimize"

    elif cmd == "/stop":
        if campaign_stop_event:
            campaign_stop_event.set()
            print("Stop signal sent. Campaign will stop after current heartbeat.")
        else:
            print("No campaign running.")

    elif cmd in ("/quit", "/exit"):
        return "quit"

    else:
        print(f"Unknown command: {cmd}. Type /help for available commands.")

    if config_changed and cfg.project_dir:
        cfg.save()

    return None


def run_campaign(cfg: HyppoConfig, stop_event: threading.Event):
    import time

    from hyppo.llm_client import LLMClient
    from hyppo.orchestrator import run_heartbeat
    from hyppo.state import WorkspaceState

    try:
        ensure_project_layout(cfg.project_dir)
        cfg.save()
    except Exception as exc:
        print(f"\nCould not initialize campaign workspace: {exc}")
        stop_event.set()
        return

    state = WorkspaceState.load_or_create(cfg.project_dir)
    client = LLMClient(cfg.provider, cfg.model)
    logger = MarkdownLogger(state.logs_dir)
    interval = cfg.heartbeat_minutes * 60

    print(f"\nCampaign started. Heartbeat every {cfg.heartbeat_minutes}m.")
    print("Type /stop or press Esc to stop, or /status to check progress.\n")

    while not stop_event.is_set():
        try:
            should_continue = run_heartbeat(state, client=client, logger=logger)
            if not should_continue:
                stop_event.set()
                break
        except Exception as exc:
            print(f"\nError in heartbeat: {exc}")
            state.save()

        if stop_event.is_set():
            break

        print(f"Sleeping {cfg.heartbeat_minutes}m until next heartbeat...")
        for _ in range(interval):
            if stop_event.is_set():
                break
            time.sleep(1)

        if not stop_event.is_set():
            state = WorkspaceState.load_or_create(cfg.project_dir)

    state.save()
    print("\nCampaign stopped.")


def load_startup_config(cwd: str | None = None) -> HyppoConfig:
    cfg = HyppoConfig()
    current_dir = os.path.abspath(cwd or os.getcwd())
    config_path = existing_project_config_path(current_dir)
    if not config_path.exists():
        return cfg

    try:
        loaded = HyppoConfig.from_project(current_dir)
    except Exception as exc:
        print(f"Warning: could not load {config_path}: {exc}")
        return cfg

    print(f"Loaded config from {config_path}")
    return loaded


class CliSession:
    def __init__(
        self,
        cfg: HyppoConfig | None = None,
        cwd: str | None = None,
        history_path: Path | None = None,
        campaign_runner=run_campaign,
        thread_factory=None,
    ):
        self.cwd = os.path.abspath(cwd or os.getcwd())
        self.cfg = cfg or load_startup_config(self.cwd)
        self.history_path = Path(history_path or DEFAULT_HISTORY_PATH)
        self.stop_event = threading.Event()
        self.campaign_thread: threading.Thread | None = None
        self._campaign_runner = campaign_runner
        self._thread_factory = thread_factory or threading.Thread

    @property
    def campaign_running(self) -> bool:
        return self.campaign_thread is not None and self.campaign_thread.is_alive()

    def start_campaign(self) -> None:
        if not self.cfg.llm_description:
            _maybe_generate_description(self.cfg, force=False)

        errors = self.cfg.validate()
        if errors:
            print("Cannot start - fix these first:")
            for error in errors:
                print(f"  - {error}")
            return

        if self.campaign_running:
            print("Campaign already running. Use /stop first.")
            return

        self.stop_event.clear()
        self.campaign_thread = self._thread_factory(
            target=self._campaign_runner,
            args=(self.cfg, self.stop_event),
            daemon=True,
        )
        self.campaign_thread.start()

    def request_stop(self) -> str | None:
        if not self.campaign_running:
            return None
        if self.stop_event.is_set():
            return "Stop already requested. Campaign will stop after current heartbeat."
        self.stop_event.set()
        return "Stop signal sent. Campaign will stop after current heartbeat."

    def process_line(self, line: str) -> bool:
        normalized = normalize_interactive_line(line, self.cfg, self.cwd)
        if not normalized:
            return True

        if not normalized.startswith("/"):
            print("Commands start with /. Type /help for available commands.")
            return True

        result = handle_command(
            normalized,
            self.cfg,
            self.stop_event if self.campaign_running else None,
        )

        if result == "quit":
            if self.campaign_running:
                print("Stopping campaign...")
                self.stop_event.set()
                self.campaign_thread.join(timeout=5)
            return False

        if result == "optimize":
            self.start_campaign()

        return True


def _build_prompt_session(cli_session: CliSession):
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.application import run_in_terminal
        from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
        from prompt_toolkit.completion import Completer, Completion
        from prompt_toolkit.history import FileHistory
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.patch_stdout import patch_stdout
        from prompt_toolkit.shortcuts import CompleteStyle
    except ImportError as exc:
        raise RuntimeError(
            "prompt_toolkit is required for the interactive CLI. "
            "Reinstall Hyppo with its dependencies."
        ) from exc

    class HyppoCompleter(Completer):
        def get_completions(self, document, complete_event):
            del complete_event
            for candidate in get_completion_candidates(
                document.text_before_cursor,
                cli_session.cfg,
                cwd=cli_session.cwd,
            ):
                yield Completion(
                    candidate.text,
                    start_position=candidate.start_position,
                    display_meta=candidate.display_meta or None,
                )

    cli_session.history_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_session = PromptSession(
        history=FileHistory(str(cli_session.history_path)),
        auto_suggest=AutoSuggestFromHistory(),
        completer=HyppoCompleter(),
    )
    bindings = KeyBindings()

    @bindings.add("escape")
    def handle_escape(event):
        message = cli_session.request_stop()
        if message:
            run_in_terminal(lambda: print(message))
            return
        if event.current_buffer.complete_state:
            event.current_buffer.cancel_completion()

    prompt_kwargs = {
        "key_bindings": bindings,
        "complete_while_typing": True,
        "complete_style": CompleteStyle.MULTI_COLUMN,
        "reserve_space_for_menu": 8,
    }
    return prompt_session, patch_stdout, prompt_kwargs


def main():
    print_banner()

    cli_session = CliSession()
    prompt_session, patch_stdout, prompt_kwargs = _build_prompt_session(cli_session)

    while True:
        try:
            with patch_stdout(raw=True):
                line = prompt_session.prompt(
                    "hyppo> ",
                    **prompt_kwargs,
                )
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not cli_session.process_line(line):
            break


if __name__ == "__main__":
    main()
