def compute_trend(losses: list[float]) -> str:
    if len(losses) < 2:
        return "insufficient_data"

    diffs = [losses[i] - losses[i - 1] for i in range(1, len(losses))]
    avg_diff = sum(diffs) / len(diffs)

    if avg_diff < -0.01:
        return "improving"
    if avg_diff < -0.001:
        return "slowly_improving"
    if avg_diff > 0.01:
        return "diverging"
    return "plateaued"


def _normalize_history(history) -> list[dict]:
    if isinstance(history, list):
        return history
    try:
        return history.to_dict("records")
    except AttributeError:
        return list(history)


def _coerce_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_run_history(run) -> list[dict]:
    scan_history = getattr(run, "scan_history", None)
    if callable(scan_history):
        return list(scan_history())

    return _normalize_history(run.history())


def _derive_time_seconds(rows: list[dict]) -> list[float | None]:
    explicit_elapsed = [_coerce_float(row.get("elapsed_time_seconds")) for row in rows]
    if any(value is not None for value in explicit_elapsed):
        return explicit_elapsed

    runtimes = [_coerce_float(row.get("_runtime")) for row in rows]
    if any(runtime is not None for runtime in runtimes):
        return runtimes

    timestamps = [_coerce_float(row.get("_timestamp")) for row in rows]
    valid_timestamps = [timestamp for timestamp in timestamps if timestamp is not None]
    if valid_timestamps:
        start = valid_timestamps[0]
        return [
            timestamp - start if timestamp is not None else None
            for timestamp in timestamps
        ]

    epochs = [_coerce_float(row.get("epoch")) for row in rows]
    if any(epoch is not None for epoch in epochs):
        return epochs

    return [None for _ in rows]


def _derive_progress_percent(row: dict, max_time_seconds: float | None, time_seconds) -> float | None:
    for key in ("progress_percent", "progress_pct", "percent", "percentage"):
        value = _coerce_float(row.get(key))
        if value is not None:
            return max(0.0, min(value, 100.0))

    progress = _coerce_float(row.get("progress"))
    if progress is not None:
        if progress <= 1.0:
            return max(0.0, min(progress * 100.0, 100.0))
        return max(0.0, min(progress, 100.0))

    if max_time_seconds and time_seconds is not None:
        return max(0.0, min((time_seconds / max_time_seconds) * 100.0, 100.0))

    return None


def empty_metrics() -> dict:
    return {
        "metric_history": [],
        "history_points": 0,
        "best_val_loss": None,
        "best_time_seconds": None,
        "best_progress_percent": None,
        "latest_val_loss": None,
        "latest_train_loss": None,
        "elapsed_time_seconds": None,
        "progress_percent": None,
        "trend": "insufficient_data",
    }


def has_metric_signal(metrics: dict) -> bool:
    return bool(
        metrics.get("metric_history")
        or metrics.get("best_val_loss") is not None
        or metrics.get("latest_val_loss") is not None
        or metrics.get("latest_train_loss") is not None
        or metrics.get("elapsed_time_seconds") is not None
        or metrics.get("progress_percent") is not None
    )


def fetch_run_metrics(wandb_run_path: str, max_time: int | None = None) -> dict:
    import wandb

    api = wandb.Api()
    run = api.run(wandb_run_path)
    rows = _read_run_history(run)
    max_time_seconds = float(max_time) * 60 if max_time else None

    if not rows:
        return empty_metrics()

    time_seconds = _derive_time_seconds(rows)
    history = []
    for row, time_value in zip(rows, time_seconds):
        val_loss = _coerce_float(row.get("val_loss"))
        train_loss = _coerce_float(row.get("train_loss"))
        progress_percent = _derive_progress_percent(row, max_time_seconds, time_value)

        if val_loss is None and train_loss is None:
            continue

        history.append(
            {
                "time_seconds": time_value,
                "progress_percent": progress_percent,
                "val_loss": val_loss,
                "train_loss": train_loss,
            }
        )

    if not history:
        return empty_metrics()

    val_points = [point for point in history if point["val_loss"] is not None]
    val_losses = [point["val_loss"] for point in val_points]
    best_point = min(val_points, key=lambda point: point["val_loss"]) if val_points else None
    latest_point = history[-1]

    return {
        "metric_history": history,
        "history_points": len(history),
        "best_val_loss": best_point["val_loss"] if best_point else None,
        "best_time_seconds": best_point["time_seconds"] if best_point else None,
        "best_progress_percent": best_point["progress_percent"] if best_point else None,
        "latest_val_loss": latest_point["val_loss"],
        "latest_train_loss": latest_point["train_loss"],
        "elapsed_time_seconds": latest_point["time_seconds"],
        "progress_percent": latest_point["progress_percent"],
        "trend": compute_trend(val_losses),
    }
