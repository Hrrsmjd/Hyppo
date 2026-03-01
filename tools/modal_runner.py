import uuid

import modal

from state import WorkspaceState, now_iso


def launch_modal_run(run_name: str, params: dict, config: dict) -> dict:
    """Spawn a Modal training function. Returns call info."""
    fn = modal.Function.from_name("hpo-agent", "train_model")
    call = fn.spawn(
        **params,
        wandb_project=config["wandb_project"],
        wandb_entity=config.get("wandb_entity"),
        run_name=run_name,
        max_epochs=config.get("max_epochs_per_run", 50),
    )
    return {"modal_call_id": call.object_id, "status": "running"}


def check_modal_run_status(modal_call_id: str) -> str:
    """Check if a Modal function call is still running."""
    call = modal.functions.FunctionCall.from_id(modal_call_id)
    try:
        call.get(timeout=0)
        return "completed"
    except TimeoutError:
        return "running"
    except Exception:
        return "failed"


def get_modal_run_result(modal_call_id: str) -> dict:
    """Get the return value from a completed Modal call."""
    call = modal.functions.FunctionCall.from_id(modal_call_id)
    return call.get()


def execute_launch_run(params: dict, state: WorkspaceState) -> dict:
    """Validate params, check concurrency, launch run, update state."""
    # Concurrency check
    if len(state.active_runs) >= state.config.get("max_concurrent_runs", 4):
        return {
            "error": "Max concurrent runs reached",
            "active": len(state.active_runs),
        }

    # Validate params against search space
    space = state.read_search_space()
    if space:
        space_params = space["parameters"]
        for param_name, value in params.items():
            if param_name not in space_params:
                return {
                    "error": f"Parameter '{param_name}' not in current search space. "
                    f"Available: {list(space_params.keys())}"
                }

    # Generate run ID before launching so we can pass it as W&B run name.
    # Append a short random suffix so retries don't collide with stale W&B runs.
    suffix = uuid.uuid4().hex[:6]
    run_id = f"run_{state.next_run_number():03d}_{suffix}"

    # Launch on Modal
    result = launch_modal_run(run_id, params, state.config)

    state.active_runs.append(
        {
            "run_id": run_id,
            "modal_call_id": result["modal_call_id"],
            "params": params,
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
