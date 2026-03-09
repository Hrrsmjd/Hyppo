# Hyppo

![Hyppo Logo](./hyppo.jpg)

Hyppo is an autonomous hyperparameter optimization CLI for training jobs that run on Modal and log metrics to Weights & Biases.

Point Hyppo at a training codebase, tell it which hyperparameters it is allowed to tune, and let an LLM run the optimization loop. Hyppo keeps all campaign state on disk, launches runs on Modal, reads metrics from W&B, and iteratively updates strategy and search space over time.

## Install

This repository is not installed from PyPI. Most users should either install directly from GitHub or clone the repo locally.

Recommended: install the CLI directly from GitHub with `uv`:

```bash
uv tool install 'git+https://github.com/Hrrsmjd/Hyppo.git'
```

Then run:

```bash
hyppo
```

If you want to clone the repo locally first:

```bash
git clone https://github.com/Hrrsmjd/Hyppo.git
cd Hyppo
```

With `uv` for local development from a clone:

```bash
uv sync --extra dev
uv run hyppo
```

With `uv` as an installed CLI tool from a local checkout:

```bash
uv tool install .
```

With `uv` from GitHub:

```bash
uv tool install 'git+https://github.com/Hrrsmjd/Hyppo.git'
```

From a local checkout:

```bash
pipx install .
```

From GitHub:

```bash
pipx install 'git+https://github.com/Hrrsmjd/Hyppo.git'
```

For development:

```bash
pip install -e ".[dev]"
```

If you prefer a manual `uv` virtualenv flow:

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
hyppo
```

## What Hyppo does

1. Reads your project and generates an LLM-facing description automatically.
2. Lets you declare the hyperparameters that are allowed to be tuned.
3. Asks the LLM to initialize a search space.
4. Launches runs on Modal.
5. Reads metrics from W&B each heartbeat.
6. Refines the search strategy until you stop the campaign or the run budget is exhausted.

## Requirements

- A training function deployed on **Modal**
- That function must accept the tuned hyperparameters plus:
  - `wandb_project`
  - `wandb_entity`
  - `run_name`
- A **Weights & Biases** project for tracking metrics
- A W&B API key available locally and inside Modal
- An API key for at least one supported LLM provider:
  - Anthropic
  - OpenAI
  - OpenRouter

## Auth and Keys

Hyppo uses three different auth surfaces:

1. An LLM provider API key for Hyppo itself. This is used locally by the CLI and does not need to be present in Modal.
2. A Weights & Biases API key both locally and inside Modal.
3. Local Modal authentication so Hyppo can launch and inspect remote runs.

### LLM provider key

You need one of:

- `ANTHROPIC_API_KEY`
- `OPENAI_API_KEY`
- `OPENROUTER_API_KEY`

You can provide it in either of these ways:

- Set the environment variable before launching Hyppo.
- Use `/provider ...` and then `/apikey ...` inside the CLI.

When you use `/apikey`, Hyppo stores the key locally in `~/.hyppo/credentials.json`.

Examples:

```bash
export ANTHROPIC_API_KEY=...
```

or inside the CLI:

```text
/provider anthropic
/apikey sk-ant-...
```

### Weights & Biases key

Hyppo reads W&B metrics locally through the W&B API, and your training job logs to W&B inside Modal. That means the W&B key needs to exist in both places:

- Local machine: use `wandb login` or set `WANDB_API_KEY`
- Modal runtime: create a Modal secret and attach it to your training function

The example training app at `examples/cifar10/modal_app.py` expects a Modal secret named `wandb-secret`. If you want to deploy that example, clone the repo first so you have the example files locally.

Example:

```bash
modal secret create wandb-secret WANDB_API_KEY=...
modal deploy examples/cifar10/modal_app.py
```

### Modal authentication

Hyppo launches Modal functions and polls their status from your local machine, so your local Modal CLI/SDK must already be authenticated before you run `/optimize`.

## First-Time Setup

1. Install Hyppo from GitHub or clone the repo locally.
2. Make sure your Modal training function is already deployed.
3. Configure your local LLM key with an environment variable or `/apikey`.
4. Configure W&B locally with `wandb login` or `WANDB_API_KEY`.
5. Create the `wandb-secret` Modal secret if your training function needs it.
6. Launch `hyppo` and point it at your training project.

## Quick Start

Launch the CLI:

```bash
hyppo
```

Or with `uv` from a repo checkout:

```bash
uv run hyppo
```

### Interactive CLI features

- Start typing a slash command to get autocomplete suggestions, such as `/pa` -> `/params`.
- Use `@` inside `/project` and `/script` to autocomplete paths, including relative paths, `../...`, `~/...`, and absolute paths.
- Use the up arrow to recall previous commands from history.
- Press `Esc` while a campaign is running to request a clean stop without exiting the CLI.

Then configure a campaign:

```text
/project ./my_training_project
/script train.py
/params learning_rate,dropout,batch_size,weight_decay
/provider anthropic
/model claude-sonnet-4-20250514
/apikey sk-...
/wandb myteam/hpo-runs
/modal my-modal-app train_model
/heartbeat 5
/max_total_runs 100
/max_concurrent_runs 4
/max_time 30
/optimize
```

If you prefer environment variables, you can launch the CLI like this:

```bash
export ANTHROPIC_API_KEY=...
export WANDB_API_KEY=...
hyppo
```

### What those commands mean

- `/project` points to the whole training codebase, not just a single script.
- `/script` points to the training script inside that project.
- `/params` is the explicit allowlist of hyperparameters the model is supposed to use.
- `/describe` appends your own notes on top of the generated project description.
- `/max_total_runs` is the total campaign budget.
- `/max_concurrent_runs` is the parallelism cap.
- `/max_time` is Hyppo's per-run time budget. In the bundled CIFAR-10
  example, Hyppo passes it through as `max_time_minutes`, so the training
  loop uses it as the effective runtime budget.

Important:
- The bundled CIFAR-10 example still has a separate higher Modal hard
  timeout as a safety cap.
- For your own custom Modal function, `/max_time` only controls real run
  time if your function accepts and uses `max_time_minutes`.

## How It Works

Every heartbeat is a fresh single-turn LLM call. Hyppo does not rely on chat history. Instead, it rebuilds the full campaign state from disk each time.

That state includes:

- config
- generated and user-supplied project description
- current search space
- active runs
- recent completed runs
- strategy notes
- historical insights

The LLM can:

- initialize the search space
- update the search space
- launch new runs
- write strategy notes

## Project Description Generation

When you set `/project`, Hyppo scans the project directory, reads relevant text files, and sends that context to the configured model to generate `llm_description`.

You can then add extra context with:

```text
/describe Use augmentation-heavy settings if validation loss plateaus early
```

Hyppo stores those two pieces separately:

- `llm_description`
- `user_description`

## Campaign State on Disk

Hyppo writes campaign state inside your project under `.hyppo/`.

```text
~/.hyppo/
├── credentials.json
└── history

your-project/
└── .hyppo/
    ├── hyppo.json
    ├── skills/
    ├── state/
    │   ├── active_runs.json
    │   ├── completed_runs.json
    │   ├── search_space.json
    │   ├── strategy.md
    │   └── all_insights.md
    └── logs/
        ├── tool_log.md
        └── llm_log.md
```

## Metrics and Logging

Hyppo tracks run metrics as structured history instead of only keeping the latest values.

State includes:

- full `metric_history`
- best validation loss
- latest validation loss
- latest training loss
- elapsed time
- progress percentage

Logs include:

- `tool_log.md`: every tool call and result
- `llm_log.md`: full prompts and model responses

Note:
- full prompt logging is useful for debugging
- it can also make logs large, especially for bigger projects

## CLI Commands

| Command | Purpose |
| --- | --- |
| `/project <path>` | Set the project directory and generate or refresh the LLM description |
| `/script <path>` | Set the training script path inside the project |
| `/describe <text>` | Append user notes to the generated description |
| `/params <list>` | Declare the hyperparameters the model is allowed to tune |
| `/provider <name>` | Set the LLM provider |
| `/model <name>` | Set the LLM model |
| `/apikey <key>` | Save the API key for the current provider |
| `/wandb <entity/project>` | Set the W&B target |
| `/modal <app> <function>` | Set the deployed Modal app and function |
| `/heartbeat <mins>` | Set the heartbeat interval |
| `/max_total_runs <n>` | Set the total run budget |
| `/max_concurrent_runs <n>` | Set the concurrent run cap |
| `/max_time <minutes>` | Set Hyppo's per-run time budget |
| `/help` | Show available commands |
| `/config` | Show the current config |
| `/status` | Show campaign status |
| `/optimize` | Start the campaign |
| `/stop` | Stop the campaign after the current heartbeat |
| `/quit` or `/exit` | Exit the CLI |

## Supported LLM Providers

| Provider | Environment Variable | Base URL |
| --- | --- | --- |
| Anthropic | `ANTHROPIC_API_KEY` | `https://api.anthropic.com/v1/` |
| OpenAI | `OPENAI_API_KEY` | `https://api.openai.com/v1` |
| OpenRouter | `OPENROUTER_API_KEY` | `https://openrouter.ai/api/v1` |

You can set LLM keys with `/apikey` or environment variables. All providers are accessed through the OpenAI SDK.

Package metadata and the `hyppo` console entry point are defined in `pyproject.toml`.
