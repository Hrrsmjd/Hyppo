"""Full heartbeat loop with mocked Modal, W&B, and Claude."""

import json
from unittest.mock import MagicMock, patch

import pytest

from state import WorkspaceState
from tools.definitions import TOOL_DEFINITIONS


@pytest.fixture
def workspace(tmp_path):
    """Create a full workspace for heartbeat testing."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    plots_dir = tmp_path / "plots"
    plots_dir.mkdir()

    config = {
        "objective": "minimize",
        "metric": "val_loss",
        "model_description": "Test transformer for sentiment analysis.",
        "training_script": "training/modal_app.py",
        "available_hyperparameters": ["learning_rate", "dropout", "num_layers"],
        "max_concurrent_runs": 4,
        "max_epochs_per_run": 10,
        "heartbeat_interval_minutes": 1,
        "wandb_project": "test-project",
    }
    (state_dir / "config.json").write_text(json.dumps(config))

    # Write a simple skill file
    (skills_dir / "tools.md").write_text("## Tools\nUse initialize_search_space to start.")

    return str(tmp_path)


def make_claude_response(tool_calls=None, text=None, stop_reason="end_turn"):
    """Build a mock Claude API response."""
    response = MagicMock()
    response.stop_reason = stop_reason
    content = []
    if text:
        block = MagicMock()
        block.type = "text"
        block.text = text
        content.append(block)
    if tool_calls:
        for tc in tool_calls:
            block = MagicMock()
            block.type = "tool_use"
            block.id = tc["id"]
            block.name = tc["name"]
            block.input = tc["input"]
            content.append(block)
    response.content = content
    return response


def test_completed_runs_truncated_in_prompt(workspace):
    """Prompt should summarize older completed runs to reduce payload size."""
    from prompt_builder import format_state_for_prompt

    state = WorkspaceState.load_or_create(workspace)
    # Add 15 completed runs — should trigger truncation (max 10 recent)
    for i in range(15):
        state.completed_runs.append({
            "run_id": f"run_{i+1:03d}",
            "best_val_loss": 0.5 - i * 0.01,
            "params": {"lr": 0.001},
        })

    text = format_state_for_prompt(state)
    assert "Older Runs (5 runs, summarized)" in text
    assert "Recent Runs" in text
    # Older runs should be one-line summaries, not full JSON
    assert "run_001: best_val_loss=" in text


def test_first_heartbeat_initializes_search_space(workspace):
    """First heartbeat should detect missing search space and Claude should initialize it."""
    from orchestrator import run_heartbeat
    from prompt_builder import build_prompt_content

    state = WorkspaceState.load_or_create(workspace)

    # Verify first-heartbeat detection in prompt
    content = build_prompt_content(state, [])
    prompt_text = content[0]["text"]
    assert "No search space has been defined yet" in prompt_text

    # Mock Claude to call initialize_search_space
    init_response = make_claude_response(
        tool_calls=[
            {
                "id": "tool_1",
                "name": "initialize_search_space",
                "input": {
                    "parameters": {
                        "learning_rate": {
                            "type": "continuous",
                            "min": 1e-5,
                            "max": 1e-2,
                            "scale": "log",
                            "notes": "Typical transformer LR",
                        },
                        "dropout": {
                            "type": "continuous",
                            "min": 0.0,
                            "max": 0.5,
                            "scale": "linear",
                            "notes": "Standard range",
                        },
                    }
                },
            }
        ],
        stop_reason="tool_use",
    )

    # After tool execution, Claude responds with text
    final_response = make_claude_response(
        text="Search space initialized with learning_rate and dropout.",
        stop_reason="end_turn",
    )

    with patch("orchestrator.client") as mock_client:
        mock_client.messages.create.side_effect = [init_response, final_response]
        run_heartbeat(state)

    # Verify search space was created
    assert state.search_space_exists()
    space = state.read_search_space()
    assert "learning_rate" in space["parameters"]
    assert "dropout" in space["parameters"]
    assert space["version"] == 1


def test_heartbeat_with_launch_run(workspace):
    """Heartbeat where Claude launches a run after search space exists."""
    from orchestrator import execute_tool_call

    state = WorkspaceState.load_or_create(workspace)

    # Set up existing search space
    from tools.search_space import execute_initialize_search_space

    execute_initialize_search_space(
        {
            "learning_rate": {"type": "continuous", "min": 1e-5, "max": 1e-2, "scale": "log", "notes": "test"},
            "dropout": {"type": "continuous", "min": 0.0, "max": 0.5, "scale": "linear", "notes": "test"},
        },
        state,
    )

    # Mock Modal launch
    with patch("tools.modal_runner.launch_modal_run") as mock_launch:
        mock_launch.return_value = {"modal_call_id": "mock_call_123", "status": "running"}

        result = execute_tool_call(
            "launch_run",
            {"params": {"learning_rate": 3e-4, "dropout": 0.1}},
            state,
        )

    assert result["status"] == "launched"
    assert result["run_id"].startswith("run_001_")
    assert len(state.active_runs) == 1


def test_heartbeat_strategy_update(workspace):
    """Heartbeat where Claude updates the strategy."""
    from orchestrator import execute_tool_call

    state = WorkspaceState.load_or_create(workspace)

    result = execute_tool_call(
        "update_strategy",
        {"content": "## Phase 1\nStarting coarse sweep with LR and dropout."},
        state,
    )
    assert result["status"] == "updated"
    assert "coarse sweep" in state.strategy


def test_full_multi_tool_heartbeat(workspace):
    """Simulate a heartbeat where Claude chains multiple tools."""
    from orchestrator import execute_tool_calls

    state = WorkspaceState.load_or_create(workspace)

    # Claude first initializes search space, then updates strategy
    first_response = make_claude_response(
        tool_calls=[
            {
                "id": "tool_1",
                "name": "initialize_search_space",
                "input": {
                    "parameters": {
                        "learning_rate": {
                            "type": "continuous",
                            "min": 1e-5,
                            "max": 1e-2,
                            "scale": "log",
                            "notes": "test",
                        },
                    }
                },
            },
            {
                "id": "tool_2",
                "name": "update_strategy",
                "input": {"content": "## Init\nDefined search space."},
            },
        ],
        stop_reason="tool_use",
    )

    final_response = make_claude_response(
        text="Done.", stop_reason="end_turn"
    )

    with patch("orchestrator.client") as mock_client:
        mock_client.messages.create.return_value = final_response
        execute_tool_calls(first_response, state)

    assert state.search_space_exists()
    assert "Defined search space" in state.strategy


def test_strip_images_from_continuation(workspace):
    """Continuation messages should not re-send images."""
    from orchestrator import _strip_images

    content = [
        {"type": "text", "text": "prompt here"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "abc"}},
        {"type": "text", "text": "more text"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "def"}},
    ]
    stripped = _strip_images(content)
    assert len(stripped) == 2
    assert all(b["type"] == "text" for b in stripped)


def test_wandb_history_normalization():
    """W&B history normalization handles both list-of-dicts and DataFrame."""
    from tools.wandb_reader import _normalize_history

    # List-of-dicts passthrough
    rows = [{"epoch": 0, "val_loss": 0.5}, {"epoch": 1, "val_loss": 0.4}]
    assert _normalize_history(rows) == rows

    # DataFrame conversion (if pandas is available)
    import pandas as pd
    df = pd.DataFrame(rows)
    normalized = _normalize_history(df)
    assert isinstance(normalized, list)
    assert len(normalized) == 2
    assert normalized[0]["epoch"] == 0


def test_update_runs_from_modal_completed(workspace):
    """Test that completed Modal runs get moved to completed_runs."""
    from orchestrator import update_runs_from_modal_and_wandb

    state = WorkspaceState.load_or_create(workspace)
    state.active_runs.append(
        {
            "run_id": "run_001",
            "modal_call_id": "call_123",
            "params": {"lr": 0.001},
            "started_at": "2026-01-01T00:00:00Z",
        }
    )

    with (
        patch("orchestrator.check_modal_run_status", return_value="completed"),
        patch("orchestrator.get_modal_run_result", return_value={"best_val_loss": 0.35}),
        patch("orchestrator.fetch_run_metrics", return_value={
            "epochs_completed": 10,
            "best_val_loss": 0.35,
            "best_epoch": 8,
            "last_3_val_losses": [0.37, 0.36, 0.35],
            "current_train_loss": 0.28,
            "trend": "improving",
        }),
    ):
        update_runs_from_modal_and_wandb(state)

    assert len(state.active_runs) == 0
    assert len(state.completed_runs) == 1
    assert state.completed_runs[0]["best_val_loss"] == 0.35
    assert state.completed_runs[0]["status"] == "completed"
