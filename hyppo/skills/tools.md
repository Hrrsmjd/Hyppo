## Available Tools

### initialize_search_space
Define the initial search space.

Input:
{
  "parameters": {
    "param_name": {
      "type": "continuous" | "categorical",
      "min": float, "max": float, "scale": "log" | "linear",
      // OR
      "options": [values],
      "notes": "Why this range"
    }
  }
}

Returns:
{"status": "created", "parameter_count": int}

### update_search_space
Modify the search space based on experimental results.

Input:
{
  "updates": {
    "param_to_narrow": {"min": new_min, "max": new_max, "notes": "reason"},
    "param_to_add": {"type": "continuous", "min": ..., "max": ..., "notes": "reason"},
    "param_to_remove": null
  },
  "changelog_entry": "Description of what changed and why"
}

Returns:
{"status": "updated", "version": int, "parameter_count": int}

Setting a parameter to null removes it from the search space.

### launch_run
Start a new training run with specified hyperparameters.

Input:
{"params": {hyperparameter dict}}

Returns:
{"run_id": str, "status": "launched"}

Parameters must match the current search space. The tool will fail if
max concurrency or total run budget has already been reached.

### update_strategy
Write your current observations and plan to the strategy file.

Input:
{"content": str}

This is persistent memory. Update it whenever you learn something new or
change your plan. Start the content with a single-line
`Insight: ...` summary.

## Tool Use Guidelines

- Before launching any new runs on a heartbeat, call `update_strategy`
  first with the evidence you are acting on and the launch plan.
- On the first heartbeat, the normal sequence is:
  `update_strategy`, `initialize_search_space`, then `launch_run`.
- On later heartbeats, the preferred order is:
  `update_strategy`, optional `update_search_space`, then `launch_run`.
- Do not call `launch_run` if the search space is missing or obviously
  stale relative to what you just learned.
- Use multiple `launch_run` calls in one heartbeat only when you have a
  clear reason to spend concurrency immediately.
- When updating the strategy, include:
  - a single-line `Insight: ...` summary first
  - what changed this heartbeat
  - what you now believe matters
  - what you plan to test next
