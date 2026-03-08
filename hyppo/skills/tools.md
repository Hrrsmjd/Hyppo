## Available Tools

### initialize_search_space
Define the initial search space. Call this on the first heartbeat
after reading the model description in the current state.
Input: {
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
Returns: {"status": "created", "parameter_count": int}

### update_search_space
Modify the search space based on experimental results. You can add
parameters, remove parameters, or change ranges.
Input: {
  "updates": {
    "param_to_narrow": {"min": new_min, "max": new_max, "notes": "reason"},
    "param_to_add": {"type": "continuous", "min": ..., "max": ..., "notes": "reason"},
    "param_to_remove": null
  },
  "changelog_entry": "Description of what changed and why"
}
Returns: {"status": "updated", "version": int, "parameter_count": int}
Setting a parameter to null removes it from the search space.

### launch_run
Start a new training run with specified hyperparameters.
Input: {"params": {hyperparameter dict}}
Parameters must match those in the current search space.
Returns: {"run_id": str, "status": "launched"}
Will fail if max concurrent runs is already reached.

### update_strategy
Write your current observations and plan to the strategy file.
Input: {"content": str}  (markdown text)
This is your persistent memory. Update it whenever you learn
something new or change your plan.
