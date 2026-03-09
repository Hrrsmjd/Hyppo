DEFAULT_METRIC = "val_loss"
DEFAULT_OBJECTIVE = "minimize"


def get_metric_name(config: dict | None) -> str:
    if not config:
        return DEFAULT_METRIC
    metric = str(config.get("metric") or DEFAULT_METRIC).strip()
    return metric or DEFAULT_METRIC


def get_objective(config: dict | None) -> str:
    if not config:
        return DEFAULT_OBJECTIVE
    objective = str(config.get("objective") or DEFAULT_OBJECTIVE).strip().lower()
    if objective not in {"minimize", "maximize"}:
        return DEFAULT_OBJECTIVE
    return objective


def is_better(candidate, incumbent, objective: str) -> bool:
    if candidate is None:
        return False
    if incumbent is None:
        return True
    if objective == "maximize":
        return candidate > incumbent
    return candidate < incumbent


def select_best_point(points: list[dict], objective: str) -> dict | None:
    best_point = None
    best_value = None
    for point in points:
        candidate = point.get("metric")
        if is_better(candidate, best_value, objective):
            best_point = point
            best_value = candidate
    return best_point


def compute_trend(values: list[float], objective: str) -> str:
    if len(values) < 2:
        return "insufficient_data"

    diffs = [values[index] - values[index - 1] for index in range(1, len(values))]
    avg_diff = sum(diffs) / len(diffs)
    if objective == "maximize":
        avg_diff *= -1

    if avg_diff < -0.01:
        return "improving"
    if avg_diff < -0.001:
        return "slowly_improving"
    if avg_diff > 0.01:
        return "diverging"
    return "plateaued"


def get_run_best_metric(run: dict, config: dict | None) -> float | None:
    metric_name = get_metric_name(config)
    candidates = ["best_metric"]
    if metric_name == "accuracy":
        candidates.extend(["best_accuracy"])
    elif metric_name == "val_loss":
        candidates.extend(["best_val_loss"])
    else:
        candidates.extend([f"best_{metric_name}"])

    for key in candidates:
        value = run.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def get_run_latest_metric(run: dict, config: dict | None) -> float | None:
    metric_name = get_metric_name(config)
    candidates = ["latest_metric"]
    if metric_name == "accuracy":
        candidates.extend(["latest_accuracy", "final_accuracy"])
    elif metric_name == "val_loss":
        candidates.extend(["latest_val_loss", "final_val_loss"])
    else:
        candidates.extend([f"latest_{metric_name}", f"final_{metric_name}"])

    for key in candidates:
        value = run.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def get_run_best_time_seconds(run: dict, config: dict | None) -> float | None:
    metric_name = get_metric_name(config)
    candidates = []
    if metric_name == "accuracy":
        candidates.append("best_accuracy_time_seconds")
    candidates.append("best_time_seconds")

    for key in candidates:
        value = run.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def get_run_best_progress_percent(run: dict, config: dict | None) -> float | None:
    metric_name = get_metric_name(config)
    candidates = []
    if metric_name == "accuracy":
        candidates.append("best_accuracy_progress_percent")
    candidates.append("best_progress_percent")

    for key in candidates:
        value = run.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None
