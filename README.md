# Hyppo

Autonomous hyperparameter optimization for Modal jobs tracked in Weights & Biases. Hyppo keeps campaign state on disk, asks an LLM what to do each heartbeat, launches runs on Modal, and refines the search space from observed metrics.

## Install

From source:

```bash
pipx install .
```

Directly from GitHub:

```bash
pipx install 'git+https://github.com/Hrrsmjd/Hyppo.git'
```

After you publish to PyPI:

```bash
pipx install hyppo
```

For development:

```bash
pip install -e ".[dev]"
```

## Prerequisites

- A training function deployed on **Modal** (see `hyppo/training/modal_app.py`)
- A **Weights & Biases** project for metric tracking
- A W&B API key available both locally and inside Modal
- An API key for at least one LLM provider (`anthropic`, `openai`, or `openrouter`)

If you use the example Modal app, copy or adapt `hyppo/training/modal_app.py`,
then create the secret and deploy your app first:

```bash
modal secret create wandb-secret WANDB_API_KEY=...
modal deploy path/to/your/modal_app.py
```

## Quick start

```bash
hyppo
```

Then configure and launch:

```
/project ./my_training
/describe ResNet-style classifier for CIFAR-10 with AdamW
/params learning_rate,dropout,batch_size,weight_decay
/provider anthropic
/model claude-sonnet-4-20250514
/apikey sk-...
/wandb myteam/hpo-runs
/modal hpo-agent train_model
/heartbeat 5
/max_runs 2
/optimize
```

Use `/script path/to/train.py` if you want to store a reference to the
source file in `hyppo.json`; actual execution targets the deployed Modal
app and function configured with `/modal`.

Hyppo will:
1. Ask the LLM to define an initial search space based on your model description
2. Launch training runs on Modal with different hyperparameter configurations
3. Poll W&B for metrics each heartbeat
4. Refine the search space based on results
5. Repeat until you `/stop`

## How it works

Each heartbeat, Hyppo sends the current campaign state to the configured
LLM. The model can initialize or update the search space, launch new
runs, and write strategy notes. All campaign state lives on disk, so a
restart can resume from `hyppo.json` and `.hyppo/state/`.

## Files Hyppo creates

```
~/.hyppo/
└── credentials.json          # API keys (never in project dir)

your-project/
├── hyppo.json                # Campaign config (safe to commit)
└── .hyppo/                   # Working directory (gitignored)
    ├── skills/               # Skill files guiding the LLM
    ├── state/
    │   ├── active_runs.json
    │   ├── completed_runs.json
    │   ├── search_space.json
    │   └── strategy.md
    └── logs/
        ├── tool_log.md       # Every tool call + result
        └── llm_log.md        # Every LLM prompt + response
```

## CLI commands

| Command | Purpose |
| --- | --- |
| `/project <path>` | Set the project directory where `hyppo.json` and `.hyppo/` live |
| `/script <path>` | Store a reference training script path in config |
| `/modal <app> <function>` | Choose the deployed Modal app and function to invoke |
| `/describe <text>` | Describe the model/task for the LLM |
| `/params <list>` | Declare candidate hyperparameters |
| `/wandb <entity/project>` | Set the W&B target |
| `/status` | Show active/completed runs and the current best metric |
| `/optimize` | Start the campaign loop |

## Supported LLM providers

| Provider | Env var | Base URL |
|----------|---------|----------|
| Anthropic | `ANTHROPIC_API_KEY` | `https://api.anthropic.com/v1/` |
| OpenAI | `OPENAI_API_KEY` | `https://api.openai.com/v1` |
| OpenRouter | `OPENROUTER_API_KEY` | `https://openrouter.ai/api/v1` |

Set keys via `/apikey` or environment variables. All providers are accessed through the OpenAI SDK.

## Publishing

Build and upload from a clean checkout:

```bash
python -m build
python -m twine check dist/*
python -m twine upload dist/*
```

The package metadata and console entry point are defined in `pyproject.toml`.
