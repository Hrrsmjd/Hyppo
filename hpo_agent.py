import argparse
import os
import shutil
import sys


TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workspace_template")


def cmd_init(args):
    workspace = args.workspace
    if os.path.exists(workspace):
        print(f"Error: {workspace} already exists.")
        sys.exit(1)

    shutil.copytree(TEMPLATE_DIR, workspace)
    # Create plots directory
    os.makedirs(os.path.join(workspace, "plots"), exist_ok=True)
    print(f"Workspace created at: {workspace}")
    print(f"Edit {os.path.join(workspace, 'state', 'config.json')} to configure your campaign.")


def cmd_run(args):
    workspace = args.workspace
    if not os.path.isdir(workspace):
        print(f"Error: workspace {workspace} does not exist. Run 'init' first.")
        sys.exit(1)

    config_path = os.path.join(workspace, "state", "config.json")
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found.")
        sys.exit(1)

    from orchestrator import main
    main(workspace)


def cli():
    parser = argparse.ArgumentParser(description="HPO Agent — autonomous hyperparameter optimization")
    subparsers = parser.add_subparsers(dest="command")

    init_parser = subparsers.add_parser("init", help="Create a new workspace from template")
    init_parser.add_argument("--workspace", required=True, help="Path to create the workspace")

    run_parser = subparsers.add_parser("run", help="Start the heartbeat loop")
    run_parser.add_argument("--workspace", required=True, help="Path to the workspace directory")

    args = parser.parse_args()

    if args.command == "init":
        cmd_init(args)
    elif args.command == "run":
        cmd_run(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    cli()
