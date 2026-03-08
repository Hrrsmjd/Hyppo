import os
import threading

from hyppo import __version__
from hyppo.config import HyppoConfig, get_api_key, save_api_key


def print_banner():
    print(f"Welcome to Hyppo {__version__} — autonomous hyperparameter optimization.")
    print("Type /help for available commands.\n")


def print_help():
    print("""
Setup commands:
  /project <path>       Set project directory for campaign state
  /script <filename>    Store reference training script path
  /describe <text>      Describe the model/task for the LLM
  /params <list>        Set hyperparameters (comma-separated)
  /provider <name>      Set LLM provider (anthropic/openai/openrouter)
  /model <name>         Set LLM model
  /apikey <key>         Set API key for current provider
  /wandb <project>      Set W&B project (optionally entity/project)
  /heartbeat <mins>     Set heartbeat interval in minutes
  /max_runs <n>         Set max concurrent runs
  /max_epochs <n>       Set max epochs per run
  /modal <app> <func>   Set Modal app and function name
  /config               Show current configuration

Campaign commands:
  /optimize             Start the optimization campaign
  /status               Show current campaign status
  /stop                 Stop campaign (if running)
  /quit                 Exit Hyppo
""")


def print_config(cfg: HyppoConfig):
    api_key = get_api_key(cfg.provider)
    key_status = f"***{api_key[-4:]}" if api_key else "NOT SET"

    print(f"""
  Project:      {cfg.project_dir or 'NOT SET'}
  Script:       {cfg.script or '(reference only)'}
  Description:  {cfg.description[:80] or 'NOT SET'}{'...' if len(cfg.description) > 80 else ''}
  Params:       {', '.join(cfg.params) if cfg.params else 'NOT SET'}
  Provider:     {cfg.provider}
  Model:        {cfg.model}
  API Key:      {key_status}
  W&B Project:  {cfg.wandb_project}
  W&B Entity:   {cfg.wandb_entity or '(default)'}
  Heartbeat:    {cfg.heartbeat_minutes}m
  Max Runs:     {cfg.max_runs}
  Max Epochs:   {cfg.max_epochs}
  Modal App:    {cfg.modal_app}
  Modal Func:   {cfg.modal_function}
""")


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

    print(f"""
  Active Runs:          {snapshot["active_runs"]}
  Completed Runs:       {snapshot["completed_runs"]}
  Best val_loss:        {best_text}
  Search Space Version: {space_version}
""")


def handle_command(line: str, cfg: HyppoConfig, campaign_stop_event: threading.Event | None) -> str | None:
    """Process a slash command. Returns 'quit' to exit, 'optimize' to start campaign."""
    parts = line.strip().split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

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
        cfg.project_dir = path
        detected = cfg.detect_script()
        if detected and not cfg.script:
            cfg.script = detected
            print(f"Project set to: {path}")
            print(f"Auto-detected training script: {detected}")
        else:
            print(f"Project set to: {path}")

    elif cmd == "/script":
        if not arg:
            print("Usage: /script <filename>")
            return None
        cfg.script = arg
        print(f"Training script: {arg}")

    elif cmd == "/describe":
        if not arg:
            print("Usage: /describe <text>")
            return None
        cfg.description = arg.strip('"').strip("'")
        print(f"Description set ({len(cfg.description)} chars)")

    elif cmd == "/params":
        if not arg:
            print("Usage: /params learning_rate,dropout,batch_size")
            return None
        cfg.params = [p.strip() for p in arg.split(",") if p.strip()]
        print(f"Parameters: {', '.join(cfg.params)}")

    elif cmd == "/provider":
        if arg not in ("anthropic", "openai", "openrouter"):
            print("Provider must be: anthropic, openai, or openrouter")
            return None
        cfg.provider = arg
        print(f"Provider: {arg}")

    elif cmd == "/model":
        if not arg:
            print("Usage: /model <model-name>")
            return None
        cfg.model = arg
        print(f"Model: {arg}")

    elif cmd == "/apikey":
        if not arg:
            print("Usage: /apikey <key>")
            return None
        save_api_key(cfg.provider, arg)
        print(f"API key saved for {cfg.provider}")

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
        print(f"W&B: {cfg.wandb_entity or '(default)'}/{cfg.wandb_project}")

    elif cmd == "/heartbeat":
        value = _parse_positive_int(arg, "/heartbeat <minutes>")
        if value is not None:
            cfg.heartbeat_minutes = value
            print(f"Heartbeat interval: {cfg.heartbeat_minutes}m")

    elif cmd == "/max_runs":
        value = _parse_positive_int(arg, "/max_runs <n>")
        if value is not None:
            cfg.max_runs = value
            print(f"Max concurrent runs: {cfg.max_runs}")

    elif cmd == "/max_epochs":
        value = _parse_positive_int(arg, "/max_epochs <n>")
        if value is not None:
            cfg.max_epochs = value
            print(f"Max epochs per run: {cfg.max_epochs}")

    elif cmd == "/modal":
        parts_modal = arg.split()
        if len(parts_modal) == 2:
            cfg.modal_app, cfg.modal_function = parts_modal
            print(f"Modal: {cfg.modal_app}::{cfg.modal_function}")
        elif len(parts_modal) == 1:
            cfg.modal_app = parts_modal[0]
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

    return None


def run_campaign(cfg: HyppoConfig, stop_event: threading.Event):
    """Run the optimization campaign in a background thread."""
    import time

    from hyppo.config import ensure_project_layout
    from hyppo.llm_client import LLMClient
    from hyppo.logger import MarkdownLogger
    from hyppo.orchestrator import run_heartbeat
    from hyppo.state import WorkspaceState

    # Save config and set up workspace
    cfg.save()
    ensure_project_layout(cfg.project_dir)

    state = WorkspaceState.load_or_create(cfg.project_dir)
    client = LLMClient(cfg.provider, cfg.model)
    logger = MarkdownLogger(state.logs_dir)
    interval = cfg.heartbeat_minutes * 60

    print(f"\nCampaign started. Heartbeat every {cfg.heartbeat_minutes}m.")
    print("Type /stop to stop, or /status to check progress.\n")

    while not stop_event.is_set():
        try:
            run_heartbeat(state, client=client, logger=logger)
        except Exception as exc:
            print(f"\nError in heartbeat: {exc}")
            state.save()

        if stop_event.is_set():
            break

        print(f"Sleeping {cfg.heartbeat_minutes}m until next heartbeat...")
        # Sleep in small increments to check stop_event
        for _ in range(interval):
            if stop_event.is_set():
                break
            time.sleep(1)

        if not stop_event.is_set():
            state = WorkspaceState.load_or_create(cfg.project_dir)

    state.save()
    print("\nCampaign stopped.")


def main():
    print_banner()

    # Try to load existing config from current directory
    cfg = HyppoConfig()
    cwd = os.getcwd()
    hyppo_json = os.path.join(cwd, "hyppo.json")
    if os.path.exists(hyppo_json):
        try:
            cfg = HyppoConfig.from_project(cwd)
            print(f"Loaded config from {hyppo_json}")
        except Exception as exc:
            print(f"Warning: could not load {hyppo_json}: {exc}")

    campaign_thread: threading.Thread | None = None
    stop_event = threading.Event()

    while True:
        try:
            line = input("hyppo> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue

        if not line.startswith("/"):
            print("Commands start with /. Type /help for available commands.")
            continue

        result = handle_command(line, cfg, stop_event if campaign_thread else None)

        if result == "quit":
            if campaign_thread and campaign_thread.is_alive():
                print("Stopping campaign...")
                stop_event.set()
                campaign_thread.join(timeout=5)
            break

        elif result == "optimize":
            errors = cfg.validate()
            if errors:
                print("Cannot start — fix these first:")
                for e in errors:
                    print(f"  - {e}")
                continue

            if campaign_thread and campaign_thread.is_alive():
                print("Campaign already running. Use /stop first.")
                continue

            stop_event.clear()
            campaign_thread = threading.Thread(
                target=run_campaign, args=(cfg, stop_event), daemon=True
            )
            campaign_thread.start()


if __name__ == "__main__":
    main()
