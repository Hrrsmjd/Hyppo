import uuid
from numbers import Real

from hyppo.state import WorkspaceState, now_iso


def launch_modal_run(run_name: str, params: dict, config: dict) -> dict:
    import modal

    app_name = config.get("modal_app_name", "hpo-agent")
    function_name = config.get("modal_function_name", "train_model")
    fn = modal.Function.from_name(app_name, function_name)
    call = fn.spawn(
        **params,
        wandb_project=config["wandb_project"],
        wandb_entity=config.get("wandb_entity"),
        run_name=run_name,
        max_epochs=config.get("max_epochs_per_run", 50),
    )
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
            "epochs_completed": 0,
            "best_val_loss": None,
            "best_epoch": None,
            "last_3_val_losses": [],
            "current_train_loss": None,
            "trend": "insufficient_data",
        }
    )
    state.save()
    return {"run_id": run_id, "status": "launched"}
