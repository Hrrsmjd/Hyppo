import json
import time

import anthropic

from prompt_builder import build_prompt_content
from state import WorkspaceState
from tools.definitions import TOOL_DEFINITIONS
from tools.modal_runner import (
    check_modal_run_status,
    execute_launch_run,
    get_modal_run_result,
)
from tools.plotter import generate_all_plots
from tools.search_space import (
    execute_initialize_search_space,
    execute_update_search_space,
)
from tools.wandb_reader import fetch_run_metrics

client = anthropic.Anthropic(max_retries=4)


def update_runs_from_modal_and_wandb(state: WorkspaceState) -> None:
    """Poll Modal for run status, pull W&B metrics, move completed runs."""
    still_active = []

    for run in state.active_runs:
        modal_call_id = run.get("modal_call_id")
        if not modal_call_id:
            still_active.append(run)
            continue

        status = check_modal_run_status(modal_call_id)

        if status == "completed":
            # Get final result and move to completed
            try:
                result = get_modal_run_result(modal_call_id)
                run.update(result if isinstance(result, dict) else {})
            except Exception as e:
                print(f"Warning: could not get result for {run['run_id']}: {e}")

            # Pull final metrics from W&B
            try:
                metrics = fetch_run_metrics(state.wandb_run_path(run["run_id"]))
                run.update(metrics)
            except Exception as e:
                print(f"Warning: could not fetch final metrics for {run['run_id']}: {e}")

            run["status"] = "completed"
            state.completed_runs.append(run)

        elif status == "failed":
            run["status"] = "failed"
            state.completed_runs.append(run)

        else:
            # Still running — update metrics from W&B
            try:
                metrics = fetch_run_metrics(state.wandb_run_path(run["run_id"]))
                run.update(metrics)
            except Exception:
                pass  # Metrics may not be available yet
            still_active.append(run)

    state._active_runs = still_active
    state.save()


def execute_tool_call(tool_name: str, tool_input: dict, state: WorkspaceState) -> dict:
    """Dispatch a single tool call to the appropriate handler."""
    if tool_name == "initialize_search_space":
        return execute_initialize_search_space(tool_input["parameters"], state)
    elif tool_name == "update_search_space":
        return execute_update_search_space(
            tool_input["updates"], tool_input["changelog_entry"], state
        )
    elif tool_name == "launch_run":
        return execute_launch_run(tool_input["params"], state)
    elif tool_name == "update_strategy":
        state.write_strategy(tool_input["content"])
        return {"status": "updated"}
    else:
        return {"error": f"Unknown tool: {tool_name}"}


def _strip_images(content: list[dict]) -> list[dict]:
    """Remove image blocks from content to reduce token count in continuations."""
    return [block for block in content if block.get("type") != "image"]


def execute_tool_calls(response, state: WorkspaceState, original_content: list[dict] | None = None) -> None:
    """Process tool calls from Claude's response, handling multi-turn tool use."""
    # Use text-only version of original content for continuations to avoid
    # re-sending base64 images (which Claude already saw in the initial call).
    continuation_content = _strip_images(original_content) if original_content else "__heartbeat__"
    messages = [{"role": "user", "content": continuation_content}]
    current_response = response

    while current_response.stop_reason == "tool_use":
        # Build assistant message from response content
        assistant_content = []
        tool_results = []

        for block in current_response.content:
            if block.type == "text":
                assistant_content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                assistant_content.append(
                    {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    }
                )
                # Execute the tool
                result = execute_tool_call(block.name, block.input, state)
                print(f"  Tool: {block.name} -> {json.dumps(result)}")
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result),
                    }
                )

        messages.append({"role": "assistant", "content": assistant_content})
        messages.append({"role": "user", "content": tool_results})

        # Continue the conversation for chained tool calls
        current_response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            tools=TOOL_DEFINITIONS,
            messages=messages,
        )

    # Print any final text
    for block in current_response.content:
        if block.type == "text" and block.text.strip():
            print(f"  Claude: {block.text}")


def run_heartbeat(state: WorkspaceState) -> bool:
    """Run a single heartbeat cycle. Returns True to continue, False to stop."""
    print(f"\n{'='*60}")
    print("Heartbeat starting...")

    # 1. Poll infrastructure and update state
    if state.active_runs:
        print("Polling Modal and W&B for run updates...")
        update_runs_from_modal_and_wandb(state)

    # 2. Generate plots for active runs
    plot_paths = []
    if state.active_runs:
        print("Generating plots...")
        try:
            plot_paths = generate_all_plots(state)
        except Exception as e:
            print(f"Warning: plot generation failed: {e}")

    # 3. Build prompt content
    content = build_prompt_content(state, plot_paths)

    # 4. Call Claude
    print("Calling Claude...")
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        tools=TOOL_DEFINITIONS,
        messages=[{"role": "user", "content": content}],
    )

    # 5. Execute tool calls
    execute_tool_calls(response, state, original_content=content)

    # 6. Save state
    state.save()

    print("Heartbeat complete.")
    return True


def main(workspace_dir: str) -> None:
    """Main loop: load state, run heartbeats with sleep interval."""
    state = WorkspaceState.load_or_create(workspace_dir)
    interval = state.config.get("heartbeat_interval_minutes", 15) * 60

    print(f"Starting HPO agent for workspace: {workspace_dir}")
    print(f"Heartbeat interval: {interval}s")

    while True:
        try:
            should_continue = run_heartbeat(state)
            if not should_continue:
                print("Campaign finished.")
                break
        except KeyboardInterrupt:
            print("\nStopped by user.")
            state.save()
            break
        except Exception as e:
            print(f"Error in heartbeat: {e}")
            state.save()

        print(f"Sleeping {interval}s until next heartbeat...")
        try:
            time.sleep(interval)
        except KeyboardInterrupt:
            print("\nStopped by user.")
            break

        # Reload state from disk in case of manual edits
        state = WorkspaceState.load_or_create(workspace_dir)
