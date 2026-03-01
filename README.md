# Hyppo

An autonomous hyperparameter optimization agent. Hyppo uses Claude to intelligently search hyperparameter space for transformer models, launching training runs on [Modal](https://modal.com) and tracking metrics via [Weights & Biases](https://wandb.ai).

The agent **defines its own search space** from a model description, then continuously refines it as experiments reveal which parameters matter and which regions are promising.

## How it works

Hyppo runs a **heartbeat loop**. Every N minutes it:

1. Reads skill files (search strategy, tool docs, metrics interpretation)
2. Reads workspace state (config, search space, active/completed runs)
3. Polls Modal for run status and W&B for training metrics
4. Generates loss curve plots for active runs
5. Sends everything to Claude in a single API call (text + images)
6. Executes Claude's tool calls (initialize/update search space, launch runs, update strategy)
7. Writes updated state back to disk

Every heartbeat is a **fresh, single-turn LLM call**. All memory lives in files on disk, not in conversation history.

## Prerequisites

- Python 3.10+
- A [Modal](https://modal.com) account with `modal` CLI set up (`modal setup`)
- A [Weights & Biases](https://wandb.ai) account
- An [Anthropic API](https://console.anthropic.com) key

## Setup

```bash
# Create and activate a conda environment (or use any virtualenv)
conda create -n hyppo python=3.12 -y
conda activate hyppo

# Install hyppo and its dependencies
cd /path/to/hyppo
pip install -e ".[dev]"

# Set environment variables
export ANTHROPIC_API_KEY="sk-ant-..."
export WANDB_API_KEY="..."
```

### Deploy the training function to Modal

The included example trains a small transformer on IMDb sentiment classification:

```bash
modal deploy training/modal_app.py
```

This registers the `train_model` function under the Modal app `hpo-agent`. You can replace this with your own training script — it just needs to accept hyperparameters as keyword arguments, log metrics to W&B, and return a results dict.

## Quick start

### 1. Create a workspace

```bash
python hpo_agent.py init --workspace ./my_campaign
```

This copies the workspace template:

```
my_campaign/
├── skills/                  # Loaded into the prompt every heartbeat
│   ├── search_strategy.md   # Phased HPO approach
│   ├── tools.md             # Tool descriptions
│   ├── metrics_interpretation.md
│   └── early_stopping.md
├── state/
│   └── config.json          # Model description, budget, settings
└── plots/                   # Loss curve PNGs (generated each heartbeat)
```

### 2. Edit the config

Open `my_campaign/state/config.json` and set:

```json
{
  "objective": "minimize",
  "metric": "val_loss",
  "model_description": "Describe your model architecture and task here. Be specific — Claude uses this to choose sensible hyperparameter ranges.",
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
  "wandb_entity": null,
  "wandb_project": "hpo-agent"
}
```

Key fields:
- **`model_description`** — Give Claude enough context to make informed decisions. Mention the architecture, dataset, task, and current training setup.
- **`available_hyperparameters`** — List the knobs your training script exposes. The agent picks which ones to tune and what ranges to use.
- **`wandb_entity`** — Set this to your W&B username or team name. If `null`, the W&B default entity is used.
- **`wandb_project`** — The W&B project where runs are logged.
- **`heartbeat_interval_minutes`** — How often the agent checks on runs and decides what to do next.

### 3. Start the agent

```bash
python hpo_agent.py run --workspace ./my_campaign
```

On the first heartbeat, the agent reads your model description and creates an initial search space. On subsequent heartbeats, it launches runs, reads metrics, updates the search space, and evolves its strategy.

Press `Ctrl+C` to stop. All state is on disk — restarting picks up where it left off.

## What the agent does

On the **first heartbeat**, with no search space defined, Claude:
- Reads the model description and available hyperparameters
- Defines an initial search space (typically 4-6 parameters with ranges based on architecture priors)
- Records its reasoning in `strategy.md`

On **subsequent heartbeats**, Claude:
- Reviews active and completed runs, including loss curve plots
- Decides whether to launch new runs, update the search space, or refine its strategy
- Narrows or expands parameter ranges based on results
- Adds or removes parameters as evidence accumulates

## Workspace state files

As the campaign runs, the agent creates and updates these files in `state/`:

| File | Purpose |
|------|---------|
| `config.json` | Campaign configuration (you set this) |
| `search_space.json` | Agent-defined parameter ranges with changelog |
| `active_runs.json` | Currently running training jobs |
| `completed_runs.json` | Finished runs with final metrics |
| `strategy.md` | Agent's observations, plan, and open questions |

All files are human-readable JSON and Markdown. You can inspect them while the campaign runs, or edit them between heartbeats to steer the agent.

## Available tools

The agent has four tools:

| Tool | What it does |
|------|-------------|
| `initialize_search_space` | Define the initial search space (first heartbeat only) |
| `update_search_space` | Narrow ranges, add/remove parameters, with changelog |
| `launch_run` | Start a training run on Modal with specified hyperparameters |
| `update_strategy` | Write observations and plans to `strategy.md` |

## Customizing skills

Skill files in `skills/` are loaded into every prompt. Edit them to change how the agent behaves:

- **`search_strategy.md`** — Phased approach (coarse sweep, narrowing, fine-tuning)
- **`tools.md`** — Tool documentation the agent references
- **`metrics_interpretation.md`** — How to read loss curves and plots
- **`early_stopping.md`** — When to kill underperforming runs

## Using your own training script

Replace `training/modal_app.py` with your own Modal app. Requirements:

1. The function must accept hyperparameters as keyword arguments, plus `wandb_project`, `wandb_entity`, `run_name`, and `max_epochs`
2. Call `wandb.init(project=wandb_project, entity=wandb_entity, id=run_name, name=run_name, config=...)` so the agent can track the run
3. Log `epoch`, `train_loss`, and `val_loss` to W&B each epoch
4. Return a dict with at least `best_val_loss` and `best_epoch`
5. Deploy with `modal deploy your_training_script.py`

Update `available_hyperparameters` in `config.json` to match the parameters your script accepts.

## Running tests

```bash
python -m pytest tests/ -v
```

Tests mock all external services (Modal, W&B, Claude API) and verify:
- State file read/write round-trips
- Search space initialization and updates
- Full heartbeat loop with tool chaining

## Project structure

```
hyppo/
├── hpo_agent.py             # CLI entry point (init, run)
├── orchestrator.py           # Heartbeat loop
├── prompt_builder.py         # Assembles skills + state + images into prompt
├── state.py                  # WorkspaceState: read/write all state files
├── tools/
│   ├── definitions.py        # Tool schemas for Claude API
│   ├── search_space.py       # Initialize + update search space
│   ├── modal_runner.py       # Modal spawn/status/result
│   ├── wandb_reader.py       # W&B metric queries
│   └── plotter.py            # Loss curve plot generation
├── training/
│   └── modal_app.py          # Example Modal training function
├── workspace_template/       # Copied on init
│   ├── skills/*.md
│   └── state/config.json
└── tests/
```
