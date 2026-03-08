# Hyppo

Hyppo is an autonomous hyperparameter optimization CLI for training jobs that run on Modal and log metrics to Weights & Biases.

Point Hyppo at a training codebase, tell it which hyperparameters it is allowed to tune, and let an LLM run the optimization loop. Hyppo keeps all campaign state on disk, launches runs on Modal, reads metrics from W&B, and iteratively updates strategy and search space over time.

## Install

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

After publishing to PyPI:

```bash
pipx install hyppo
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

If you use the example Modal app in `hyppo/training/modal_app.py`, create the W&B secret and deploy it first:

```bash
modal secret create wandb-secret WANDB_API_KEY=...
modal deploy path/to/your/modal_app.py
```

## Quick Start

Launch the CLI:

```bash
hyppo
```

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

### What those commands mean

- `/project` points to the whole training codebase, not just a single script.
- `/script` points to the training script inside that project.
- `/params` is the explicit allowlist of hyperparameters the model is supposed to use.
- `/describe` appends your own notes on top of the generated project description.
- `/max_total_runs` is the total campaign budget.
- `/max_concurrent_runs` is the parallelism cap.
- `/max_time` is Hyppo's per-run time budget for reasoning and progress normalization.

Important:
- `/max_time` does not automatically change your Modal function timeout.
- The actual hard runtime limit still needs to be configured on your deployed `@app.function(timeout=...)`.

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
└── credentials.json

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
| `/config` | Show the current config |
| `/status` | Show campaign status |
| `/optimize` | Start the campaign |
| `/stop` | Stop the campaign after the current heartbeat |

## Supported LLM Providers

| Provider | Environment Variable | Base URL |
| --- | --- | --- |
| Anthropic | `ANTHROPIC_API_KEY` | `https://api.anthropic.com/v1/` |
| OpenAI | `OPENAI_API_KEY` | `https://api.openai.com/v1` |
| OpenRouter | `OPENROUTER_API_KEY` | `https://openrouter.ai/api/v1` |

You can set keys with `/apikey` or environment variables. All providers are accessed through the OpenAI SDK.

## Publishing

Build and upload from a clean checkout:

```bash
python -m build
python -m twine check dist/*
python -m twine upload dist/*
```

Package metadata and the `hyppo` console entry point are defined in `pyproject.toml`.
