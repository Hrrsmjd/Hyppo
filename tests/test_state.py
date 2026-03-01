import json
import os
import tempfile

import pytest

from state import WorkspaceState


@pytest.fixture
def workspace(tmp_path):
    """Create a minimal workspace with config.json."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()

    config = {
        "objective": "minimize",
        "metric": "val_loss",
        "model_description": "Test model",
        "training_script": "train.py",
        "available_hyperparameters": ["learning_rate", "dropout"],
        "max_concurrent_runs": 4,
        "max_epochs_per_run": 10,
        "heartbeat_interval_minutes": 1,
        "wandb_project": "test-project",
    }
    (state_dir / "config.json").write_text(json.dumps(config))
    return str(tmp_path)


def test_load_or_create(workspace):
    state = WorkspaceState.load_or_create(workspace)
    assert state.active_runs == []
    assert state.completed_runs == []
    assert state.config["metric"] == "val_loss"


def test_search_space_lifecycle(workspace):
    state = WorkspaceState.load_or_create(workspace)
    assert not state.search_space_exists()
    assert state.search_space is None

    space = {
        "version": 1,
        "parameters": {"lr": {"type": "continuous", "min": 1e-5, "max": 1e-2}},
        "changelog": [],
    }
    state.write_search_space(space)
    assert state.search_space_exists()

    loaded = state.read_search_space()
    assert loaded["parameters"]["lr"]["min"] == 1e-5


def test_active_runs_round_trip(workspace):
    state = WorkspaceState.load_or_create(workspace)
    state.active_runs.append(
        {"run_id": "run_001", "params": {"lr": 0.001}, "best_val_loss": 0.5}
    )
    state.save_active_runs()

    # Reload
    state2 = WorkspaceState.load_or_create(workspace)
    assert len(state2.active_runs) == 1
    assert state2.active_runs[0]["run_id"] == "run_001"


def test_completed_runs_round_trip(workspace):
    state = WorkspaceState.load_or_create(workspace)
    state.completed_runs.append(
        {"run_id": "run_001", "best_val_loss": 0.3}
    )
    state.save_completed_runs()

    state2 = WorkspaceState.load_or_create(workspace)
    assert len(state2.completed_runs) == 1
    assert state2.completed_runs[0]["best_val_loss"] == 0.3


def test_next_run_number(workspace):
    state = WorkspaceState.load_or_create(workspace)
    assert state.next_run_number() == 1

    state.active_runs.append({"run_id": "run_001"})
    assert state.next_run_number() == 2

    state.completed_runs.append({"run_id": "run_005"})
    assert state.next_run_number() == 6


def test_strategy_round_trip(workspace):
    state = WorkspaceState.load_or_create(workspace)
    assert state.strategy == ""

    state.write_strategy("## Phase 1\nDoing coarse sweep.")
    assert "coarse sweep" in state.strategy


def test_best_completed_val_loss(workspace):
    state = WorkspaceState.load_or_create(workspace)
    assert state.best_completed_val_loss() is None

    state.completed_runs.extend([
        {"run_id": "run_001", "best_val_loss": 0.5},
        {"run_id": "run_002", "best_val_loss": 0.3},
        {"run_id": "run_003", "best_val_loss": 0.4},
    ])
    assert state.best_completed_val_loss() == 0.3


def test_best_completed_val_loss_with_failed_runs(workspace):
    """Failed runs have best_val_loss=None — must not crash min()."""
    state = WorkspaceState.load_or_create(workspace)
    state.completed_runs.extend([
        {"run_id": "run_001", "best_val_loss": 0.5},
        {"run_id": "run_002", "best_val_loss": None, "status": "failed"},
        {"run_id": "run_003", "best_val_loss": 0.3},
    ])
    assert state.best_completed_val_loss() == 0.3


def test_best_completed_val_loss_all_failed(workspace):
    """If all runs failed (all None), should return None."""
    state = WorkspaceState.load_or_create(workspace)
    state.completed_runs.extend([
        {"run_id": "run_001", "best_val_loss": None, "status": "failed"},
        {"run_id": "run_002", "status": "failed"},
    ])
    assert state.best_completed_val_loss() is None


def test_find_active_run(workspace):
    state = WorkspaceState.load_or_create(workspace)
    state.active_runs.append({"run_id": "run_001", "params": {}})
    state.active_runs.append({"run_id": "run_002", "params": {}})

    assert state.find_active_run("run_001")["run_id"] == "run_001"
    assert state.find_active_run("run_003") is None


def test_save_flushes_all(workspace):
    state = WorkspaceState.load_or_create(workspace)
    state.active_runs.append({"run_id": "run_001"})
    state.completed_runs.append({"run_id": "run_002", "best_val_loss": 0.4})
    state.save()

    state2 = WorkspaceState.load_or_create(workspace)
    assert len(state2.active_runs) == 1
    assert len(state2.completed_runs) == 1
