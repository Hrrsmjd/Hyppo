import uuid
from numbers import Real

from hyppo.state import WorkspaceState, now_iso


def _spawn_kwargs(run_name: str, params: dict, config: dict) -> dict:
    kwargs = {
        **params,
        "wandb_project": config["wandb_project"],
        "wandb_entity": config.get("wandb_entity"),
        "run_name": run_name,
    }
    max_time = config.get("max_time")
    if isinstance(max_time, Real) and not isinstance(max_time, bool) and max_time > 0:
        kwargs["max_time_minutes"] = int(max_time)
    return kwargs


def launch_modal_run(run_name: str, params: dict, config: dict) -> dict:
    import modal

    app_name = config.get("modal_app_name", "hpo-agent")
    function_name = config.get("modal_function_name", "train_model")
    fn = modal.Function.from_name(app_name, function_name)
    kwargs = _spawn_kwargs(run_name, params, config)
    try:
        call = fn.spawn(**kwargs)
    except TypeError as exc:
        if "max_time_minutes" not in kwargs:
            raise
        message = str(exc)
        if "max_time_minutes" not in message and "unexpected keyword" not in message:
            raise
        kwargs.pop("max_time_minutes", None)
        call = fn.spawn(**kwargs)
    return {"modal_call_id": call.object_id, "status": "running"}


def _is_number(value) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def check_modal_run_status(modal_call_id: str) -> dict:
    import modal

    call = modal.functions.FunctionCall.from_id(modal_call_id)
    try:
        call.get(timeout=0)
        return {"status": "completed"}
    except TimeoutError:
        return {"status": "running"}
    except Exception as exc:
        message = str(exc) or exc.__class__.__name__
        lowered = message.lower()
        transient_markers = (
            "timeout",
            "temporar",
            "rate limit",
            "connection",
            "network",
            "unavailable",
            "503",
            "429",
        )
        if any(marker in lowered for marker in transient_markers):
            return {"status": "unknown", "error": message}
        return {"status": "failed", "error": message}


def get_modal_run_result(modal_call_id: str) -> dict:
    import modal

    call = modal.functions.FunctionCall.from_id(modal_call_id)
    return call.get()


def _validate_param(param_name: str, value, definition: dict) -> str | None:
    param_type = definition.get("type")
    if param_type == "continuous":
        if not _is_number(value):
            return f"Parameter '{param_name}' must be numeric"
        minimum = definition.get("min")
        maximum = definition.get("max")
        if minimum is not None and value < minimum:
            return f"Parameter '{param_name}' below minimum {minimum}"
        if maximum is not None and value > maximum:
            return f"Parameter '{param_name}' above maximum {maximum}"
    elif param_type == "categorical":
        options = definition.get("options") or []
        if options and value not in options:
            return f"Parameter '{param_name}' must be one of {options}"
    else:
        return f"Parameter '{param_name}' has unsupported type '{param_type}'"
    return None


def execute_launch_run(params: dict, state: WorkspaceState) -> dict:
    if len(state.active_runs) >= state.config.get("max_concurrent_runs", 4):
        return {"error": "Max concurrent runs reached", "active": len(state.active_runs)}

    if state.max_total_runs_reached():
        return {
            "error": "Max total runs reached",
            "total_runs_started": state.total_runs_started(),
        }

    space = state.read_search_space()
    if space:
        space_params = space["parameters"]
        for param_name, value in params.items():
            if param_name not in space_params:
                return {
                    "error": f"Parameter '{param_name}' not in current search space. "
                    f"Available: {list(space_params.keys())}"
                }
            validation_error = _validate_param(param_name, value, space_params[param_name])
            if validation_error:
                return {"error": validation_error}

    suffix = uuid.uuid4().hex[:6]
    run_id = f"run_{state.next_run_number():03d}_{suffix}"
    result = launch_modal_run(run_id, params, state.config)

    state.active_runs.append(
        {
            "run_id": run_id,
            "modal_call_id": result["modal_call_id"],
            "params": params,
            "status": "running",
            "started_at": now_iso(),
            "metric_history": [],
            "history_points": 0,
            "best_metric": None,
            "best_val_loss": None,
            "best_accuracy": None,
            "best_time_seconds": None,
            "best_progress_percent": None,
            "latest_metric": None,
            "latest_val_loss": None,
            "latest_accuracy": None,
            "latest_train_loss": None,
            "elapsed_time_seconds": None,
            "progress_percent": None,
            "trend": "insufficient_data",
        }
    )
    state.save()
    return {"run_id": run_id, "status": "launched"}
