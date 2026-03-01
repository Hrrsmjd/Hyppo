# HPO Agent: Engineering Plan (v3)

## Overview

An autonomous hyperparameter optimization agent that uses Claude to intelligently search hyperparameter space for transformer models. The agent launches training runs on **Modal**, tracks metrics via **Weights & Biases**, and manages state through **flat files in a workspace directory** — inspired by the OpenClaw architecture pattern.

The agent **defines its own search space** based on the model architecture and task description, then continuously refines it as experiments reveal which regions are promising and which parameters matter.

### Core Philosophy

Each heartbeat is a **fresh, single-turn LLM call**. All memory lives in files on disk, not in conversation history. There is no context accumulation and no context compression. The agent reads the current state, makes a decision, acts, writes the updated state, and the call is done.

This is the same pattern OpenClaw uses for its general-purpose assistant: persistent files + heartbeat timer + LLM with tools. We specialize it for ML hyperparameter optimization.

### Prior Art

- **AgentHPO** (Liu et al., 2024 — [arXiv:2402.01881](https://arxiv.org/abs/2402.01881)): Demonstrated that an LLM agent can match or beat human-selected hyperparameters across 12 ML tasks by iteratively proposing and evaluating configurations. A key finding was that the LLM brings useful priors about what hyperparameter ranges make sense for different architectures — this is exactly the capability we leverage by having the agent define and refine its own search space.
- **OpenClaw**: General-purpose AI agent using heartbeat scheduling and Markdown/YAML state files. We adopt its architectural pattern and specialize for ML.
- **AWS LLM-augmented HPO**: Uses LLMs to analyze gradient norms and suggest architecture changes mid-training. Complementary — could be added as an advanced skill later.

---

## Architecture

```
Every N minutes (heartbeat):

    ┌──────────────────────────────────────────┐
    │           Orchestrator Loop               │
    │                                           │
    │  1. Read workspace/skills/*.md            │
    │  2. Read workspace/state/*.json + *.md    │
    │  3. Poll Modal + W&B → update state files │
    │  4. Generate metric plots (PNG)           │
    │  5. Build prompt from skills + state       │
    │  6. Single-turn Claude call (with images) │
    │  7. Execute tool calls                    │
    │  8. Write updated state back to disk      │
    │  9. Check termination → generate report   │
    └──────────────────────────────────────────┘
         │              │              │
    ┌────▼────┐   ┌─────▼─────┐  ┌────▼────┐
    │  Modal  │   │   W&B     │  │  Claude  │
    │  (GPU   │   │  (metric  │  │  API     │
    │  runs)  │   │  storage) │  │          │
    └─────────┘   └───────────┘  └──────────┘
```

**Key principle:** Every heartbeat is a fresh call. State lives in files, not conversation history.

---

## Workspace Structure

```
workspace/
├── skills/                        # Loaded into prompt every heartbeat
│   ├── search_strategy.md         # Phased HPO approach
│   ├── tools.md                   # Tool descriptions and usage
│   ├── metrics_interpretation.md  # How to read metrics + plots
│   └── early_stopping.md          # When to kill a run
│
├── state/                         # Read and written every heartbeat
│   ├── config.json                # Model description, objective, budget limits
│   ├── search_space.json          # Agent-defined; created and updated by agent
│   ├── active_runs.json           # Currently running jobs
│   ├── completed_runs.json        # All finished runs with results
│   ├── killed_runs.json           # Killed runs with reasons
│   ├── budget.json                # {total, spent, remaining}
│   └── strategy.md                # Agent's current thinking and plan
│
├── plots/                         # Generated each heartbeat, included as images
│   ├── run_042_loss.png
│   ├── run_043_loss.png
│   └── comparison.png             # Overlay of top runs
│
└── reports/
    └── final_report.md            # Generated at end of campaign
```

### State Files in Detail

**config.json** — Set by the user at campaign start. Describes the model and task, but does NOT define the search space — the agent does that:

```json
{
  "objective": "minimize",
  "metric": "val_loss",
  "model_description": "A transformer encoder for text classification on the IMDb sentiment dataset. 6-layer baseline with learned positional embeddings, vocab size 30k, sequence length 512. Currently trains for 20 epochs with AdamW, cosine LR schedule.",
  "training_script": "training/modal_app.py",
  "available_hyperparameters": [
    "learning_rate", "batch_size", "num_layers", "hidden_dim",
    "num_heads", "dropout", "weight_decay", "warmup_steps",
    "label_smoothing", "gradient_clip_norm", "lr_schedule"
  ],
  "max_concurrent_runs": 4,
  "max_epochs_per_run": 50,
  "heartbeat_interval_minutes": 15,
  "modal_gpu": "A100",
  "wandb_project": "hpo-agent"
}
```

The `model_description` field gives Claude enough context to make informed decisions about what search space to define. The `available_hyperparameters` field tells the agent which knobs the training script actually exposes — the agent picks which ones to tune and what ranges to use.

**search_space.json** — Created by the agent on the first heartbeat, updated as the campaign progresses:

```json
{
  "version": 3,
  "created_at": "2026-02-27T10:00:00Z",
  "last_updated": "2026-02-27T14:30:00Z",
  "parameters": {
    "learning_rate": {
      "type": "continuous",
      "min": 1e-5,
      "max": 1e-2,
      "scale": "log",
      "notes": "Initial range based on typical transformer LRs"
    },
    "num_layers": {
      "type": "categorical",
      "options": [4, 6, 8],
      "notes": "Excluding 12+ — dataset too small for deep models (observed in runs 003, 007)"
    },
    "hidden_dim": {
      "type": "categorical",
      "options": [256, 512],
      "notes": "Narrowed from [256, 512, 768] — 768 shows no benefit over 512"
    },
    "dropout": {
      "type": "continuous",
      "min": 0.05,
      "max": 0.25,
      "scale": "linear",
      "notes": "Narrowed from [0.0, 0.5] — sweet spot is 0.1-0.2"
    },
    "weight_decay": {
      "type": "continuous",
      "min": 0.01,
      "max": 0.1,
      "scale": "log",
      "notes": "Added in v2 — initial experiments showed regularization helps"
    }
  },
  "changelog": [
    {
      "version": 1,
      "timestamp": "2026-02-27T10:00:00Z",
      "description": "Initial search space: lr, num_layers, hidden_dim, dropout, batch_size, warmup_steps"
    },
    {
      "version": 2,
      "timestamp": "2026-02-27T12:15:00Z",
      "description": "Added weight_decay. Removed batch_size (32 vs 64 makes no difference). Narrowed dropout to [0.0, 0.3]."
    },
    {
      "version": 3,
      "timestamp": "2026-02-27T14:30:00Z",
      "description": "Narrowed dropout to [0.05, 0.25]. Removed num_layers=12 option. Removed hidden_dim=768."
    }
  ]
}
```

The `changelog` is important — it gives the agent (on future heartbeats) a record of why the search space looks the way it does. The `notes` field on each parameter serves a similar purpose at the individual parameter level.

**active_runs.json** — Updated by orchestrator each heartbeat:

```json
[
  {
    "run_id": "run_042",
    "params": {"lr": 3e-4, "layers": 6, "dropout": 0.1, "hidden_dim": 512, "weight_decay": 0.03},
    "started_at": "2026-02-27T10:30:00Z",
    "epochs_completed": 15,
    "best_val_loss": 0.342,
    "best_epoch": 12,
    "last_3_val_losses": [0.351, 0.348, 0.345],
    "current_train_loss": 0.298,
    "trend": "slowly_improving"
  }
]
```

**strategy.md** — Written by Claude, read back on next heartbeat:

```markdown
## Current Phase
Narrowing — focusing on the region around run_042's config.

## Key Observations
- Learning rates above 1e-3 consistently diverge (runs 003, 007, 015, 022)
- 6-layer models outperform 12-layer — dataset likely too small for deeper nets
- Dropout sweet spot is 0.1-0.2; above 0.3 hurts consistently
- Batch size 32 vs 64 makes negligible difference; removed from search space
- Added weight_decay after noticing overfitting gap in several runs

## Search Space Rationale
Started with 6 parameters. Removed batch_size (insensitive) and narrowed
dropout + num_layers based on early results. Added weight_decay after
seeing train-val gap suggesting regularization would help.

## Current Plan
Testing warmup schedule variations: run_042 used 500 steps, now trying
200 and 1000 with everything else held constant.

## Open Questions
- Should I add label_smoothing? The train-val gap is smaller now with
  weight_decay, but might still benefit from it.
- Warmup interacts with LR — may need to co-vary them.
```

---

## Visual Metrics

On each heartbeat, the orchestrator generates matplotlib plots and sends them to Claude as images alongside the numeric summaries.

### What gets plotted

**Per active run — loss curve plot:**
- Validation loss (primary) and training loss over epochs
- Horizontal dashed line at the best val_loss across all completed runs
- Title: run_id + key params

**Comparison plot (when 3+ completed runs exist):**
- Overlay of validation loss curves for the top 5 runs
- Lets Claude see which configs converge faster, which plateau earlier

### Implementation

```python
import matplotlib.pyplot as plt
import wandb
import base64

def generate_run_plot(run_id: str, project: str, best_global_loss: float) -> str:
    """Generate a loss curve PNG for a run. Returns file path."""
    api = wandb.Api()
    run = api.run(f"{project}/{run_id}")
    history = run.history(keys=["val_loss", "train_loss", "epoch"])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history["epoch"], history["val_loss"], label="val_loss", linewidth=2)
    ax.plot(history["epoch"], history["train_loss"], label="train_loss",
            linewidth=1, alpha=0.5)
    ax.axhline(y=best_global_loss, color="green", linestyle="--",
               label=f"best overall ({best_global_loss:.4f})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{run_id}")
    ax.legend()

    path = f"workspace/plots/{run_id}_loss.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return path
```

Plots are included in the Claude API call as image content blocks:

```python
content = [{"type": "text", "text": prompt_text}]

for plot_path in plot_paths:
    with open(plot_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    content.append({
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": img_b64}
    })
```

### Cost consideration

Each plot image is roughly 500-1500 tokens. For 4 active runs + 1 comparison plot, that's ~5K-7.5K extra tokens per heartbeat — negligible. If running many concurrent runs, only plot active runs plus top completed runs.

---

## Skill Files

### skills/search_strategy.md

```markdown
## Hyperparameter Search Strategy

You are optimizing hyperparameters for a transformer model. Your budget
is limited, so be strategic.

### Initialization (First Heartbeat)
Read the model description and available hyperparameters in config.json.
Based on your knowledge of the architecture and task, define an initial
search space using the `initialize_search_space` tool. Consider:
- What are typical good ranges for this architecture?
- Which hyperparameters are most likely to impact performance?
- Start broader than you think necessary — you can narrow later.
Don't include every available parameter. Start with the 4-6 most
impactful ones and add more if needed.

### Phase 1 — Coarse Sweep (first 20% of budget)
Launch diverse configurations spanning the search space. Vary multiple
parameters at once. Goal: map the landscape, find promising regions,
and identify which parameters matter.

After the coarse sweep, update the search space:
- Remove parameters that don't seem to affect performance
- Narrow ranges around promising regions
- Add new parameters if you suspect they'd help (e.g., add weight_decay
  if you're seeing overfitting)

### Phase 2 — Narrowing (middle 60% of budget)
Focus on regions around the best runs. Vary 1-2 parameters at a time.
Continue refining the search space as you learn more. This is where
most of the gains happen.

### Phase 3 — Fine-tuning (final 20% of budget)
Small perturbations of the top 2-3 configurations. Confirm results
are stable. Search space should be tight around the best-known values.

### Search Space Management
- Always document why you're changing the search space (use the notes
  and changelog fields)
- If the best result is at the boundary of a range, expand that range
- If a large region of the space consistently performs poorly, narrow it
- Be willing to add parameters mid-campaign if evidence suggests they
  matter (e.g., add label_smoothing if you're overfitting)
- Be willing to remove parameters that don't affect results — this
  focuses the remaining budget on what matters
```

### skills/tools.md

```markdown
## Available Tools

### initialize_search_space
Define the initial search space. Call this on the first heartbeat
after reading the model description in config.json.
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
Input: hyperparameter dict (must match parameters in current search space)
Returns: {"run_id": str, "status": "launched", "estimated_cost": float}
Will fail if budget is insufficient or max concurrent runs reached.

### kill_run
Terminate a running job.
Input: {"run_id": str, "reason": str}
Returns: {"run_id": str, "status": "killed"}
Always provide a reason — it's saved for future reference.

### update_strategy
Write your current observations and plan to the strategy file.
Input: {"content": str}  (markdown text)
This is your persistent memory. Update it whenever you learn
something new or change your plan.

### finish_campaign
Signal that optimization is complete. Triggers final report generation.
Input: {"summary": str, "recommended_config": dict}
Call when confident in your recommendation or budget is exhausted.
```

### skills/metrics_interpretation.md

```markdown
## Reading Loss Curve Plots

Each heartbeat includes loss curve plots for active runs. Use these
to assess training dynamics:

- **Smooth downward curve**: Healthy training, let it continue
- **Oscillating but trending down**: LR may be slightly high, but
  could still converge — monitor for 2-3 more heartbeats
- **Sharp spike then recovery**: Possible data issue or LR too high;
  if it recovers, fine; if spikes repeat, consider killing
- **Flat plateau for many epochs**: Model has converged or is stuck;
  check if val_loss is competitive with best runs
- **Train loss dropping but val_loss rising**: Overfitting — consider
  killing, or note that regularization parameters (dropout, weight_decay,
  label_smoothing) might need to be added to the search space
- **Loss exploded (NaN or very large)**: Kill immediately

A green dashed line shows the best val_loss across all completed runs.
Runs significantly above this line after many epochs should be killed.

## Using Plots for Search Space Decisions

The plots can inform search space updates:
- If many runs show overfitting (train-val gap), consider adding
  regularization parameters
- If runs with high LR all show oscillation, narrow the LR upper bound
- If loss curves look identical across different values of a parameter,
  that parameter probably doesn't matter — remove it
```

### skills/early_stopping.md

```markdown
## When to Kill a Run

Kill a run early to free budget for more promising experiments.

### Kill immediately if:
- Loss has exploded (NaN, very large values, or visible in the plot)
- Val_loss is 2x or more worse than the current best after 30%+ of
  max epochs

### Consider killing if:
- Val_loss has plateaued for 5+ epochs with no improvement, and it's
  significantly worse than the best known run
- Training loss is dropping but val_loss is rising (overfitting)
- The loss curve shows repeated large oscillations

### Keep running if:
- Loss is still actively decreasing, even if not yet competitive
- You're in the fine-tuning phase and need to confirm a result
- The run is testing a genuinely novel region of the search space
```

---

## Tool Implementations

### Tool: initialize_search_space

```python
def execute_initialize_search_space(parameters: dict, state: WorkspaceState) -> dict:
    if state.search_space_exists():
        return {"error": "Search space already initialized. Use update_search_space."}

    search_space = {
        "version": 1,
        "created_at": now_iso(),
        "last_updated": now_iso(),
        "parameters": parameters,
        "changelog": [{
            "version": 1,
            "timestamp": now_iso(),
            "description": f"Initial search space: {', '.join(parameters.keys())}"
        }]
    }
    state.write_search_space(search_space)
    return {"status": "created", "parameter_count": len(parameters)}
```

### Tool: update_search_space

```python
def execute_update_search_space(
    updates: dict, changelog_entry: str, state: WorkspaceState
) -> dict:
    space = state.read_search_space()
    if not space:
        return {"error": "No search space exists. Use initialize_search_space first."}

    for param, value in updates.items():
        if value is None:
            # Remove parameter
            space["parameters"].pop(param, None)
        elif param in space["parameters"]:
            # Update existing parameter
            space["parameters"][param].update(value)
        else:
            # Add new parameter
            space["parameters"][param] = value

    space["version"] += 1
    space["last_updated"] = now_iso()
    space["changelog"].append({
        "version": space["version"],
        "timestamp": now_iso(),
        "description": changelog_entry
    })
    state.write_search_space(space)

    return {"status": "updated", "version": space["version"],
            "parameter_count": len(space["parameters"])}
```

### Tool: launch_run

```python
import modal

def execute_launch_run(params: dict, state: WorkspaceState) -> dict:
    # Budget check (hard enforcement)
    estimated_cost = estimate_run_cost(params, state.config)
    if state.budget["remaining"] < estimated_cost:
        return {"error": "Insufficient budget", "remaining": state.budget["remaining"]}

    # Concurrency check
    if len(state.active_runs) >= state.config["max_concurrent_runs"]:
        return {"error": "Max concurrent runs reached",
                "active": len(state.active_runs)}

    # Validate params against current search space
    validation_error = validate_params_against_search_space(
        params, state.read_search_space()
    )
    if validation_error:
        return {"error": validation_error}

    # Launch on Modal
    fn = modal.Function.from_name("hpo-agent", "train_model")
    call = fn.spawn(**params, wandb_project=state.config["wandb_project"])

    run_id = f"run_{state.next_run_number():03d}"

    state.active_runs.append({
        "run_id": run_id,
        "modal_id": call.object_id,
        "params": params,
        "started_at": now_iso(),
    })
    state.budget["spent"] += estimated_cost
    state.budget["remaining"] -= estimated_cost
    state.save()

    return {"run_id": run_id, "status": "launched", "estimated_cost": estimated_cost}
```

### Tool: kill_run

```python
def execute_kill_run(run_id: str, reason: str, state: WorkspaceState) -> dict:
    run = state.find_active_run(run_id)
    if not run:
        return {"error": f"No active run with id {run_id}"}

    modal.functions.FunctionCall.from_id(run["modal_id"]).cancel()

    run["killed_at"] = now_iso()
    run["reason"] = reason
    state.killed_runs.append(run)
    state.active_runs.remove(run)
    state.save()

    return {"run_id": run_id, "status": "killed"}
```

### Tool: update_strategy

```python
def execute_update_strategy(content: str, state: WorkspaceState) -> dict:
    with open("workspace/state/strategy.md", "w") as f:
        f.write(content)
    return {"status": "updated"}
```

---

## Orchestrator (Main Loop)

```python
import time
import anthropic

client = anthropic.Anthropic()

def run_heartbeat(state: WorkspaceState):
    """Single heartbeat: read state, call Claude, execute actions, write state."""

    # 1. Poll infrastructure and update state files
    update_runs_from_modal_and_wandb(state)

    # 2. Generate plots for active runs
    plot_paths = generate_all_plots(state)

    # 3. Build the prompt
    skills_text = load_all_skills("workspace/skills/")
    state_text = format_state_for_prompt(state)
    prompt = f"{skills_text}\n\n---\n\n{state_text}"

    # 4. Build content blocks (text + images)
    content = [{"type": "text", "text": prompt}]
    for path in plot_paths:
        content.append(make_image_block(path))

    # 5. Single-turn Claude call
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        tools=TOOL_DEFINITIONS,
        messages=[{"role": "user", "content": content}],
    )

    # 6. Execute tool calls (may involve multiple rounds if Claude
    #    chains tools, e.g., update_search_space then launch_run)
    execute_tool_calls(response, state)

    # 7. Check termination
    if should_terminate(state, response):
        generate_final_report(state)
        return False

    return True

def main():
    state = WorkspaceState.load_or_create("workspace/")
    interval = state.config["heartbeat_interval_minutes"] * 60

    while True:
        should_continue = run_heartbeat(state)
        if not should_continue:
            break
        time.sleep(interval)
```

### First Heartbeat Behavior

On the first heartbeat, `search_space.json` doesn't exist yet. The prompt builder detects this and adds a special instruction:

```
No search space has been defined yet. This is your first heartbeat.
Read the model description and available hyperparameters in config.json,
then use the initialize_search_space tool to define your initial search
space before launching any runs.
```

Claude then reads the model description, applies its knowledge of transformer architectures, and creates the initial search space. On subsequent heartbeats, it reads the existing search space and decides whether to update it.

---

## Implementation Plan

### Phase 1: MVP — Single Run + Dynamic Search Space (Days 1-4)

**Goal:** Agent creates its own search space and runs sequential experiments.

**Build:**
- [ ] Workspace directory structure and state file read/write (`state.py`)
- [ ] `initialize_search_space` and `update_search_space` tools
- [ ] `launch_run` tool with Modal integration
- [ ] W&B metric reader returning summarized dict
- [ ] Basic loss curve plot generation (single run)
- [ ] Heartbeat loop: read state → build prompt → call Claude → execute → write state
- [ ] First-heartbeat detection (no search_space.json → prompt agent to initialize)
- [ ] Skill files: `tools.md`, `search_strategy.md`
- [ ] `config.json` with model description and available hyperparameters

**Skip for now:** Parallel runs, comparison plots, kill logic, budget tracking, crash recovery.

**Test:**
- Provide a model description for a small transformer
- Fast training (~5 min per run, 5-10 epochs)
- Verify: Claude reads model description and creates a sensible initial search space
- Verify: Search space reflects reasonable priors for the architecture (e.g., log-scale for LR)
- Verify: Claude launches runs, reads results, proposes next config
- Verify: After 5+ runs, Claude updates the search space (narrowing or adding parameters)
- Verify: Loss curve plots are included and Claude references them in reasoning

**Success criteria:** Claude creates a reasonable search space from the model description, runs 5-10 sequential trials, updates the search space at least once based on results, and finds a config better than random search.

---

### Phase 2: Parallel Runs + Kill Logic (Days 5-7)

**Goal:** Multiple concurrent runs with early termination.

**Build:**
- [ ] Support for multiple active runs in state
- [ ] `kill_run` tool with Modal cancellation
- [ ] `early_stopping.md` skill file
- [ ] Per-run and comparison loss curve plots
- [ ] Concurrency limit enforcement in `launch_run`
- [ ] `metrics_interpretation.md` skill file

**Test:**
- Launch 3-4 simultaneous runs
- Verify: Claude reasons about multiple active runs and their plots
- Verify: Claude kills underperforming runs based on curve shapes
- Verify: State files stay consistent

**Success criteria:** Claude manages 3-4 parallel runs, kills at least one early based on plot analysis, and uses freed budget for better experiments.

---

### Phase 3: Strategy Persistence + Budget (Days 8-10)

**Goal:** Durable observations, budget awareness, and search space evolution.

**Build:**
- [ ] `update_strategy` tool
- [ ] `strategy.md` loaded into prompt each heartbeat
- [ ] `budget.json` tracking with hard enforcement in orchestrator
- [ ] Run cost estimation
- [ ] Budget visibility in state summary
- [ ] Search space validation in `launch_run` (params must be within current space)

**Test:**
- Run a campaign long enough for 15-20 trials
- Verify: `strategy.md` accumulates useful observations
- Verify: `search_space.json` evolves meaningfully (parameters added, removed, or narrowed with documented reasons in changelog)
- Verify: Claude adapts strategy based on its own prior observations
- Verify: Budget enforcement works — orchestrator refuses launches when exhausted
- Verify: Claude proactively manages budget ("5 runs left, switching to fine-tuning")

**Success criteria:** Agent runs for 4+ hours, search space changes at least 3 times with sensible changelog entries, strategy is coherent across heartbeats, and budget is respected.

---

### Phase 4: Crash Recovery + Robustness (Days 11-12)

**Goal:** System survives failures gracefully.

**Build:**
- [ ] On startup, detect existing workspace and resume from state files
- [ ] Handle Modal job failures (timeout, OOM, crash)
- [ ] Handle W&B query failures (retry with backoff)
- [ ] Stuck-run detection: if a run hasn't reported metrics in 2x expected time, mark failed
- [ ] "No improvement for N trials" detection — force phase transition or suggest termination
- [ ] Structured logging for orchestrator debugging

**Test:**
- Kill orchestrator mid-campaign, restart — verify seamless resume
- Simulate Modal job crash — verify agent handles it and moves on
- Run a campaign where the agent exhausts promising options — verify graceful termination

**Success criteria:** Kill and restart the orchestrator 3 times during a campaign with no data loss or behavioral degradation.

---

### Phase 5: Report Generation + CLI Polish (Days 13-15)

**Build:**
- [ ] `finish_campaign` tool
- [ ] Final report generator: ranked results table, search space evolution timeline, key observations, recommended config, loss curves of top runs
- [ ] CLI: `python hpo_agent.py init` creates workspace from template
- [ ] CLI: `python hpo_agent.py run --workspace ./workspace` starts campaign
- [ ] CLI: `python hpo_agent.py status --workspace ./workspace` prints current state
- [ ] README with setup instructions

**Test:**
- Full end-to-end campaign on a real transformer tuning task
- Verify report includes search space evolution history
- Verify report is accurate, readable, and actionable

**Success criteria:** A researcher can `init`, edit `config.json` with their model description, `run`, walk away for a day, and come back to a report that includes the best hyperparameters, how the search space evolved, and why.

---

## File Structure (Codebase)

```
hpo-agent/
├── hpo_agent.py                 # CLI entry point (init, run, status)
├── orchestrator.py              # Heartbeat loop
├── tools/
│   ├── __init__.py
│   ├── definitions.py           # Tool schemas for Claude API
│   ├── search_space.py          # initialize + update search space
│   ├── modal_runner.py          # Modal spawn/kill/status
│   ├── wandb_reader.py          # W&B queries + metric summaries
│   ├── plotter.py               # Matplotlib plot generation
│   └── budget.py                # Cost estimation + tracking
├── state.py                     # WorkspaceState: read/write all state files
├── prompt_builder.py            # Assemble skills + state + images into prompt
├── report.py                    # Final report generation
├── workspace_template/          # Copied on `init`
│   ├── skills/
│   │   ├── search_strategy.md
│   │   ├── tools.md
│   │   ├── metrics_interpretation.md
│   │   └── early_stopping.md
│   └── state/
│       └── config.json          # Template with placeholders
├── training/
│   └── modal_app.py             # Example Modal training function
└── tests/
    ├── test_tools.py
    ├── test_state.py
    ├── test_search_space.py
    ├── test_plotter.py
    └── test_mock_heartbeat.py   # Full loop with mocked Modal/W&B
```

---

## Key Design Decisions

1. **Every heartbeat is a fresh call.** No conversation history, no context compression. All memory lives in workspace files.

2. **Agent defines and evolves the search space.** The user describes the model and lists available knobs. The agent decides what to tune, what ranges to use, and continuously refines based on results. The changelog provides full auditability.

3. **Visual metrics alongside numeric.** Loss curve plots give Claude the gestalt view; numeric summaries provide precision. Both are sent every heartbeat.

4. **Skills as editable Markdown files.** Change search strategy, add tools, or adjust early stopping heuristics by editing a file.

5. **Strategy.md is the agent's scratchpad.** Observations, plans, and open questions persist across heartbeats without conversation history.

6. **Budget enforcement is hybrid.** Hard limits in the orchestrator, intelligent allocation by Claude.

7. **State files are human-readable and git-friendly.** Monitor the campaign live, version-control your workspace, debug by reading JSON and Markdown.

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Agent defines unreasonable search space | Validate ranges in tool (e.g., LR > 0, layers > 0). Skill file provides guidance. User can manually edit search_space.json between heartbeats if needed. |
| Agent keeps expanding search space, wasting budget | Skill file guides phased narrowing. Budget enforcement prevents runaway spending. |
| Agent removes a parameter prematurely | Changelog preserves history. Agent can re-add parameters. User can review search_space.json. |
| Claude proposes hyperparameters outside search space | `launch_run` validates against current search_space.json, returns error. |
| Modal job hangs | Hard timeouts on Modal functions; orchestrator marks as failed. |
| Orchestrator crashes | All state is on disk; restart resumes from last written state. |
| Agent gets stuck (no improvement) | Detect "N trials without improvement" and prompt phase transition or termination. |

---

## Estimated Costs (Per Campaign)

| Item | Estimate | Notes |
|------|----------|-------|
| Claude API | $5-25 | ~100-200 heartbeats × ~6K tokens each (text + images) |
| Modal compute | $50-500+ | Depends on GPU type, run count, duration |
| W&B | Free tier | Sufficient for most campaigns |
| Matplotlib | Free | Local plot generation |

Claude API cost is negligible compared to GPU compute.
