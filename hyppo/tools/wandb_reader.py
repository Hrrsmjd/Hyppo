def compute_trend(losses: list[float]) -> str:
    if len(losses) < 2:
        return "insufficient_data"

    diffs = [losses[i] - losses[i - 1] for i in range(1, len(losses))]
    avg_diff = sum(diffs) / len(diffs)

    if avg_diff < -0.01:
        return "improving"
    elif avg_diff < -0.001:
        return "slowly_improving"
    elif avg_diff > 0.01:
        return "diverging"
    else:
        return "plateaued"


def _normalize_history(history) -> list[dict]:
    if isinstance(history, list):
        return history
    try:
        return history.to_dict("records")
    except AttributeError:
        return list(history)


def _extract_column(rows: list[dict], key: str) -> list:
    return [r[key] for r in rows if r.get(key) is not None]


def fetch_run_metrics(wandb_run_path: str) -> dict:
    import wandb

    api = wandb.Api()
    run = api.run(wandb_run_path)
    raw_history = run.history(keys=["val_loss", "train_loss", "epoch"])
    rows = _normalize_history(raw_history)

    if not rows:
        return {
            "epochs_completed": 0,
            "best_val_loss": None,
            "best_epoch": None,
            "last_3_val_losses": [],
            "current_train_loss": None,
            "trend": "insufficient_data",
        }

    valid = [r for r in rows if r.get("val_loss") is not None]
    val_losses = [r["val_loss"] for r in valid]
    epochs = [r["epoch"] for r in valid]

    train_losses = [r["train_loss"] for r in valid if r.get("train_loss") is not None]
    if not train_losses:
        train_losses = _extract_column(rows, "train_loss")

    all_epochs = _extract_column(rows, "epoch")
    epochs_completed = len(all_epochs)

    best_val_loss = min(val_losses) if val_losses else None
    best_idx = val_losses.index(best_val_loss) if best_val_loss is not None else None
    best_epoch = int(epochs[best_idx]) if best_idx is not None else None

    last_3 = val_losses[-3:] if len(val_losses) >= 3 else val_losses

    return {
        "epochs_completed": epochs_completed,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "last_3_val_losses": last_3,
        "current_train_loss": train_losses[-1] if train_losses else None,
        "trend": compute_trend(last_3),
    }
