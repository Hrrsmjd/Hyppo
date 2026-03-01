import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb

from state import WorkspaceState
from tools.wandb_reader import _normalize_history


def generate_run_plot(
    wandb_run_path: str,
    run_id: str,
    best_global_loss: float | None,
    output_dir: str,
) -> str:
    """Generate a loss curve PNG for a single run. Returns file path.

    Args:
        wandb_run_path: Full W&B path, e.g. 'entity/project/run_id'.
        run_id: Our synthetic run ID (for titles and filenames).
        best_global_loss: Best val_loss across completed runs (drawn as dashed line).
        output_dir: Directory to write the PNG to.
    """
    api = wandb.Api()
    run = api.run(wandb_run_path)
    rows = _normalize_history(run.history(keys=["val_loss", "train_loss", "epoch"]))

    epochs = [r["epoch"] for r in rows if r.get("epoch") is not None]
    val_losses = [r.get("val_loss") for r in rows if r.get("epoch") is not None]
    train_losses = [r.get("train_loss") for r in rows if r.get("epoch") is not None]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, val_losses, label="val_loss", linewidth=2)
    ax.plot(epochs, train_losses, label="train_loss", linewidth=1, alpha=0.5)

    if best_global_loss is not None:
        ax.axhline(
            y=best_global_loss,
            color="green",
            linestyle="--",
            label=f"best overall ({best_global_loss:.4f})",
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{run_id}")
    ax.legend()

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{run_id}_loss.png")
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_all_plots(state: WorkspaceState) -> list[str]:
    """Generate loss curve plots for all active runs."""
    if not state.active_runs:
        return []

    best_loss = state.best_completed_val_loss()
    output_dir = str(state.plots_dir)
    paths = []

    for run in state.active_runs:
        run_id = run["run_id"]
        try:
            path = generate_run_plot(
                state.wandb_run_path(run_id),
                run_id,
                best_loss,
                output_dir,
            )
            paths.append(path)
        except Exception as e:
            print(f"Warning: could not generate plot for {run_id}: {e}")

    return paths
