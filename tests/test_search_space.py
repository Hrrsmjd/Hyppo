import json

import pytest

from state import WorkspaceState
from tools.search_space import (
    execute_initialize_search_space,
    execute_update_search_space,
)


@pytest.fixture
def workspace(tmp_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()

    config = {
        "objective": "minimize",
        "metric": "val_loss",
        "model_description": "Test model",
        "available_hyperparameters": ["learning_rate", "dropout", "weight_decay"],
        "max_concurrent_runs": 4,
        "wandb_project": "test",
    }
    (state_dir / "config.json").write_text(json.dumps(config))
    return str(tmp_path)


def test_initialize_search_space(workspace):
    state = WorkspaceState.load_or_create(workspace)
    params = {
        "learning_rate": {
            "type": "continuous",
            "min": 1e-5,
            "max": 1e-2,
            "scale": "log",
            "notes": "Typical transformer LR range",
        },
        "dropout": {
            "type": "continuous",
            "min": 0.0,
            "max": 0.5,
            "scale": "linear",
            "notes": "Standard dropout range",
        },
    }
    result = execute_initialize_search_space(params, state)
    assert result["status"] == "created"
    assert result["parameter_count"] == 2

    # Verify on disk
    space = state.read_search_space()
    assert space["version"] == 1
    assert "learning_rate" in space["parameters"]
    assert len(space["changelog"]) == 1


def test_initialize_twice_fails(workspace):
    state = WorkspaceState.load_or_create(workspace)
    params = {"lr": {"type": "continuous", "min": 1e-5, "max": 1e-2, "scale": "log", "notes": "test"}}
    execute_initialize_search_space(params, state)

    result = execute_initialize_search_space(params, state)
    assert "error" in result


def test_update_search_space_narrow(workspace):
    state = WorkspaceState.load_or_create(workspace)
    params = {
        "learning_rate": {
            "type": "continuous",
            "min": 1e-5,
            "max": 1e-2,
            "scale": "log",
            "notes": "Initial range",
        },
        "dropout": {
            "type": "continuous",
            "min": 0.0,
            "max": 0.5,
            "scale": "linear",
            "notes": "Initial range",
        },
    }
    execute_initialize_search_space(params, state)

    result = execute_update_search_space(
        updates={"learning_rate": {"min": 1e-4, "max": 5e-3, "notes": "Narrowed based on results"}},
        changelog_entry="Narrowed LR range based on first 5 runs",
        state=state,
    )
    assert result["status"] == "updated"
    assert result["version"] == 2

    space = state.read_search_space()
    assert space["parameters"]["learning_rate"]["min"] == 1e-4
    assert space["parameters"]["learning_rate"]["max"] == 5e-3
    assert space["parameters"]["learning_rate"]["scale"] == "log"  # Preserved
    assert len(space["changelog"]) == 2


def test_update_search_space_remove(workspace):
    state = WorkspaceState.load_or_create(workspace)
    execute_initialize_search_space(
        {
            "lr": {"type": "continuous", "min": 1e-5, "max": 1e-2, "scale": "log", "notes": "test"},
            "dropout": {"type": "continuous", "min": 0.0, "max": 0.5, "scale": "linear", "notes": "test"},
        },
        state,
    )

    result = execute_update_search_space(
        updates={"dropout": None},
        changelog_entry="Removed dropout — no effect on results",
        state=state,
    )
    assert result["parameter_count"] == 1
    assert "dropout" not in state.read_search_space()["parameters"]


def test_update_search_space_add(workspace):
    state = WorkspaceState.load_or_create(workspace)
    execute_initialize_search_space(
        {"lr": {"type": "continuous", "min": 1e-5, "max": 1e-2, "scale": "log", "notes": "test"}},
        state,
    )

    result = execute_update_search_space(
        updates={
            "weight_decay": {
                "type": "continuous",
                "min": 0.01,
                "max": 0.1,
                "scale": "log",
                "notes": "Added due to overfitting",
            }
        },
        changelog_entry="Added weight_decay",
        state=state,
    )
    assert result["parameter_count"] == 2
    assert "weight_decay" in state.read_search_space()["parameters"]


def test_update_without_init_fails(workspace):
    state = WorkspaceState.load_or_create(workspace)
    result = execute_update_search_space(
        updates={"lr": {"min": 1e-4}},
        changelog_entry="test",
        state=state,
    )
    assert "error" in result
