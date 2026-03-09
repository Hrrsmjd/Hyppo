import json
import time

from hyppo.llm_client import LLMClient
from hyppo.logger import MarkdownLogger
from hyppo.prompt_builder import build_prompt
from hyppo.state import WorkspaceState, now_iso
from hyppo.tools.definitions import TOOL_DEFINITIONS
from hyppo.tools.modal_runner import (
    check_modal_run_status,
    execute_launch_run,
    get_modal_run_result,
)
from hyppo.tools.search_space import (
    execute_initialize_search_space,
    execute_update_search_space,
)
from hyppo.tools.wandb_reader import empty_metrics, fetch_run_metrics, has_metric_signal


def _hydrate_metrics(run: dict, metrics: dict) -> None:
    if has_metric_signal(metrics):
        run.update(metrics)
        return

    if run.get("final_val_loss") is not None:
        fallback_point = {
            "time_seconds": run.get("best_time_seconds") or run.get("elapsed_time_seconds"),
            "progress_percent": run.get("best_progress_percent") or run.get("progress_percent"),
            "val_loss": run.get("final_val_loss"),
            "train_loss": run.get("latest_train_loss"),
        }
        run.update(
            {
                "metric_history": [fallback_point],
                "history_points": 1,
                "best_val_loss": run.get("best_val_loss") or run.get("final_val_loss"),
                "best_time_seconds": run.get("best_time_seconds"),
                "best_progress_percent": run.get("best_progress_percent"),
                "latest_val_loss": run.get("latest_val_loss") or run.get("final_val_loss"),
                "latest_train_loss": run.get("latest_train_loss"),
                "elapsed_time_seconds": run.get("elapsed_time_seconds"),
                "progress_percent": run.get("progress_percent") or run.get("best_progress_percent"),
                "trend": run.get("trend", "insufficient_data"),
            }
        )
        return

    for key, value in empty_metrics().items():
        run.setdefault(key, value)


def _safe_fetch_run_metrics(state: WorkspaceState, run: dict, source: str) -> None:
    try:
        metrics = fetch_run_metrics(
            state.wandb_run_path(run["run_id"]),
            max_time=state.config.get("max_time"),
        )
        _hydrate_metrics(run, metrics)
    except Exception as exc:
        run["last_error"] = {
            "source": source,
            "message": str(exc),
            "timestamp": now_iso(),
        }
        print(f"Warning: could not fetch metrics for {run['run_id']}: {exc}")


def _needs_metric_backfill(run: dict) -> bool:
    return not has_metric_signal(
        {
            "metric_history": run.get("metric_history"),
            "best_val_loss": run.get("best_val_loss"),
            "latest_val_loss": run.get("latest_val_loss"),
            "latest_train_loss": run.get("latest_train_loss"),
            "elapsed_time_seconds": run.get("elapsed_time_seconds"),
            "progress_percent": run.get("progress_percent"),
        }
    )


def backfill_completed_run_metrics(state: WorkspaceState) -> None:
    updated = False
    for run in state.completed_runs:
        if not _needs_metric_backfill(run):
            continue
        _safe_fetch_run_metrics(state, run, "wandb_backfill_metrics")
        updated = True

    if updated:
        state.save()


def update_runs_from_modal_and_wandb(state: WorkspaceState) -> None:
    still_active = []

    for run in state.active_runs:
        modal_call_id = run.get("modal_call_id")
        if not modal_call_id:
            still_active.append(run)
            continue

        status_info = check_modal_run_status(modal_call_id)
        status = status_info["status"]
        if status == "completed":
            try:
                result = get_modal_run_result(modal_call_id)
                run.update(result if isinstance(result, dict) else {})
            except Exception as exc:
                run["last_error"] = {
                    "source": "modal_result",
                    "message": str(exc),
                    "timestamp": now_iso(),
                }
                print(f"Warning: could not get result for {run['run_id']}: {exc}")

            _safe_fetch_run_metrics(state, run, "wandb_final_metrics")
            run["status"] = "completed"
            run["finished_at"] = now_iso()
            state.completed_runs.append(run)
        elif status == "failed":
            run["status"] = "failed"
            run["finished_at"] = now_iso()
            if status_info.get("error"):
                run["last_error"] = {
                    "source": "modal_status",
                    "message": status_info["error"],
                    "timestamp": now_iso(),
                }
            state.completed_runs.append(run)
        else:
            if status_info.get("error"):
                run["last_error"] = {
                    "source": "modal_status",
                    "message": status_info["error"],
                    "timestamp": now_iso(),
                }
            _safe_fetch_run_metrics(state, run, "wandb_metrics")
            still_active.append(run)

    state.replace_active_runs(still_active)
    state.save()


def _validate_tool_input(tool_name: str, tool_input) -> str | None:
    if not isinstance(tool_input, dict):
        return f"Invalid input for {tool_name}: expected a JSON object"

    required_fields = {
        "initialize_search_space": ["parameters"],
        "update_search_space": ["updates", "changelog_entry"],
        "launch_run": ["params"],
        "update_strategy": ["content"],
    }

    if tool_name not in required_fields:
        return None

    missing = [field for field in required_fields[tool_name] if field not in tool_input]
    if missing:
        missing_fields = ", ".join(missing)
        return f"Invalid input for {tool_name}: missing required field(s): {missing_fields}"

    return None


def execute_tool_call(tool_name: str, tool_input: dict, state: WorkspaceState) -> dict:
    validation_error = _validate_tool_input(tool_name, tool_input)
    if validation_error:
        return {"error": validation_error}

    if tool_name == "initialize_search_space":
        return execute_initialize_search_space(tool_input["parameters"], state)
    if tool_name == "update_search_space":
        return execute_update_search_space(tool_input["updates"], tool_input["changelog_entry"], state)
    if tool_name == "launch_run":
        return execute_launch_run(tool_input["params"], state)
    if tool_name == "update_strategy":
        state.write_strategy(tool_input["content"])
        return {"status": "updated"}
    return {"error": f"Unknown tool: {tool_name}"}


def _extract_text(response) -> str:
    msg = response.choices[0].message
    return msg.content or ""


def execute_tool_calls(
    response,
    state: WorkspaceState,
    client: LLMClient,
    logger: MarkdownLogger | None = None,
    original_prompt: str | None = None,
) -> None:
    messages = [{"role": "user", "content": original_prompt or "__heartbeat__"}]
    current_response = response

    while current_response.choices[0].finish_reason == "tool_calls":
        msg = current_response.choices[0].message

        assistant_msg = {"role": "assistant"}
        if msg.content:
            assistant_msg["content"] = msg.content
        if msg.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ]
        messages.append(assistant_msg)

        for tc in msg.tool_calls:
            try:
                tool_input = json.loads(tc.function.arguments)
            except json.JSONDecodeError as exc:
                tool_input = {"raw_arguments": tc.function.arguments}
                result = {"error": f"Invalid tool arguments: {exc.msg}"}
            else:
                try:
                    result = execute_tool_call(tc.function.name, tool_input, state)
                except Exception as exc:
                    result = {"error": f"Tool execution failed for {tc.function.name}: {exc}"}
            print(f"  Tool: {tc.function.name} -> {json.dumps(result)}")
            if logger:
                logger.log_tool(tc.function.name, tool_input, result)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result),
                }
            )

        current_response = client.chat(messages=messages, tools=TOOL_DEFINITIONS)
        if logger:
            logger.log_response(_extract_text(current_response), current_response.choices[0].finish_reason)

    final_text = _extract_text(current_response).strip()
    if final_text:
        print(f"  LLM: {final_text}")


def run_heartbeat(
    state: WorkspaceState,
    client: LLMClient | None = None,
    logger: MarkdownLogger | None = None,
) -> bool:
    print(f"\n{'=' * 60}")
    print("Heartbeat starting...")

    state.reload_config()
    backfill_completed_run_metrics(state)

    if state.active_runs:
        print("Polling Modal and W&B for run updates...")
        update_runs_from_modal_and_wandb(state)

    prompt = build_prompt(state)

    if logger:
        logger.new_heartbeat()
        logger.log_prompt(prompt)

    if client is None:
        config = state.config
        client = LLMClient(config["llm_provider"], config["llm_model"])

    print("Calling LLM...")
    response = client.chat(
        messages=[{"role": "user", "content": prompt}],
        tools=TOOL_DEFINITIONS,
    )
    if logger:
        logger.log_response(_extract_text(response), response.choices[0].finish_reason)

    execute_tool_calls(response, state, client=client, logger=logger, original_prompt=prompt)
    state.save()

    should_continue = not (state.max_total_runs_reached() and not state.active_runs)
    if not should_continue:
        print("Run budget exhausted and no active runs remain.")

    print("Heartbeat complete.")
    return should_continue


def main(project_dir: str) -> None:
    state = WorkspaceState.load_or_create(project_dir)

    config = state.config
    client = LLMClient(config["llm_provider"], config["llm_model"])
    logger = MarkdownLogger(state.logs_dir)
    interval = config.get("heartbeat_interval_minutes", 5) * 60

    print(f"Starting Hyppo for project: {project_dir}")
    print(f"Heartbeat interval: {interval}s")

    while True:
        try:
            should_continue = run_heartbeat(state, client=client, logger=logger)
            if not should_continue:
                print("Campaign finished.")
                break
        except KeyboardInterrupt:
            print("\nStopped by user.")
            state.save()
            break
        except Exception as exc:
            print(f"Error in heartbeat: {exc}")
            state.save()

        print(f"Sleeping {interval}s until next heartbeat...")
        try:
            time.sleep(interval)
        except KeyboardInterrupt:
            print("\nStopped by user.")
            break

        state = WorkspaceState.load_or_create(project_dir)
