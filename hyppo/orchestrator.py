import json
import time

from hyppo.config import load_project_config
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
from hyppo.tools.wandb_reader import fetch_run_metrics


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
            try:
                run.update(fetch_run_metrics(state.wandb_run_path(run["run_id"])))
            except Exception as exc:
                run["last_error"] = {
                    "source": "wandb_final_metrics",
                    "message": str(exc),
                    "timestamp": now_iso(),
                }
                print(f"Warning: could not fetch final metrics for {run['run_id']}: {exc}")
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
            try:
                run.update(fetch_run_metrics(state.wandb_run_path(run["run_id"])))
            except Exception as exc:
                run["last_error"] = {
                    "source": "wandb_metrics",
                    "message": str(exc),
                    "timestamp": now_iso(),
                }
            still_active.append(run)

    state.replace_active_runs(still_active)
    state.save()


def execute_tool_call(tool_name: str, tool_input: dict, state: WorkspaceState) -> dict:
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
    """Extract text content from an OpenAI chat completion response."""
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

        # Build assistant message for conversation history
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

        # Execute each tool call and build tool result messages
        for tc in msg.tool_calls:
            try:
                tool_input = json.loads(tc.function.arguments)
            except json.JSONDecodeError as exc:
                tool_input = {"raw_arguments": tc.function.arguments}
                result = {"error": f"Invalid tool arguments: {exc.msg}"}
            else:
                result = execute_tool_call(tc.function.name, tool_input, state)
            print(f"  Tool: {tc.function.name} -> {json.dumps(result)}")
            if logger:
                logger.log_tool(tc.function.name, tool_input, result)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result),
            })

        # Continue conversation
        current_response = client.chat(messages=messages, tools=TOOL_DEFINITIONS)
        if logger:
            logger.log_response(_extract_text(current_response), current_response.choices[0].finish_reason)

    # Print any final text
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
    print("Heartbeat complete.")
    return True


def main(project_dir: str) -> None:
    state = WorkspaceState.load_or_create(project_dir)

    config = load_project_config(project_dir)
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
