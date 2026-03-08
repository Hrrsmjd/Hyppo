"""Full heartbeat loop with mocked LLM (OpenAI format)."""

import json
import tempfile
import unittest
from dataclasses import dataclass, field
from unittest.mock import patch

from hyppo.config import save_project_config
from hyppo.orchestrator import run_heartbeat, update_runs_from_modal_and_wandb
from hyppo.prompt_builder import build_prompt, format_state_for_prompt
from hyppo.state import WorkspaceState
from hyppo.tools.modal_runner import execute_launch_run


@dataclass
class MockFunction:
    name: str
    arguments: str


@dataclass
class MockToolCall:
    id: str
    type: str = "function"
    function: MockFunction = None


@dataclass
class MockMessage:
    content: str | None = None
    tool_calls: list[MockToolCall] | None = None


@dataclass
class MockChoice:
    message: MockMessage = None
    finish_reason: str = "stop"


@dataclass
class MockResponse:
    choices: list[MockChoice] = field(default_factory=list)


def make_tool_response(tool_calls: list[dict]) -> MockResponse:
    mock_tcs = []
    for tc in tool_calls:
        mock_tcs.append(
            MockToolCall(
                id=tc["id"],
                function=MockFunction(
                    name=tc["name"],
                    arguments=json.dumps(tc["input"]),
                ),
            )
        )
    return MockResponse(
        choices=[
            MockChoice(
                message=MockMessage(tool_calls=mock_tcs),
                finish_reason="tool_calls",
            )
        ]
    )


def make_text_response(text: str) -> MockResponse:
    return MockResponse(
        choices=[
            MockChoice(
                message=MockMessage(content=text),
                finish_reason="stop",
            )
        ]
    )


@dataclass
class FakeClient:
    responses: list

    def chat(self, messages, tools=None):
        return self.responses.pop(0)


class HeartbeatTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        save_project_config(
            self.temp_dir.name,
            {
                "model_description": "Test transformer for sentiment analysis.",
                "available_hyperparameters": [
                    "learning_rate",
                    "dropout",
                    "num_layers",
                ],
                "wandb_project": "test-project",
                "llm_provider": "anthropic",
                "llm_model": "claude-sonnet-4-20250514",
            },
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_completed_runs_truncated_in_prompt(self):
        state = WorkspaceState.load_or_create(self.temp_dir.name)
        for index in range(15):
            state.completed_runs.append(
                {
                    "run_id": f"run_{index + 1:03d}",
                    "best_val_loss": 0.5 - index * 0.01,
                    "params": {"lr": 0.001},
                    "status": "completed",
                }
            )

        text = format_state_for_prompt(state)
        self.assertIn("Older Completed Runs", text)
        self.assertIn("Recent Completed Runs", text)

    def test_first_heartbeat_instructions_present(self):
        state = WorkspaceState.load_or_create(self.temp_dir.name)
        prompt = build_prompt(state)
        self.assertIn("No search space has been defined yet", prompt)

    def test_run_heartbeat_initializes_search_space(self):
        state = WorkspaceState.load_or_create(self.temp_dir.name)
        client = FakeClient(
            responses=[
                make_tool_response(
                    [
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
                                        "notes": "Typical range",
                                    }
                                }
                            },
                        }
                    ]
                ),
                make_text_response("Initialized search space."),
            ]
        )

        run_heartbeat(state, client=client)
        self.assertTrue(state.search_space_exists())
        self.assertIn("learning_rate", state.read_search_space()["parameters"])

    def test_launch_run_validates_param_range(self):
        state = WorkspaceState.load_or_create(self.temp_dir.name)
        state.write_search_space(
            {
                "version": 1,
                "parameters": {
                    "learning_rate": {
                        "type": "continuous",
                        "min": 1e-5,
                        "max": 1e-2,
                        "scale": "log",
                        "notes": "Typical range",
                    }
                },
                "changelog": [],
            }
        )

        with patch("hyppo.tools.modal_runner.launch_modal_run") as mock_launch:
            mock_launch.return_value = {
                "modal_call_id": "call_123",
                "status": "running",
            }
            result = execute_launch_run({"learning_rate": 0.1}, state)

        self.assertIn("error", result)

    def test_launch_run_rejects_non_numeric_continuous_value(self):
        state = WorkspaceState.load_or_create(self.temp_dir.name)
        state.write_search_space(
            {
                "version": 1,
                "parameters": {
                    "learning_rate": {
                        "type": "continuous",
                        "min": 1e-5,
                        "max": 1e-2,
                        "notes": "Typical range",
                    }
                },
                "changelog": [],
            }
        )

        result = execute_launch_run({"learning_rate": "fast"}, state)

        self.assertEqual(result["error"], "Parameter 'learning_rate' must be numeric")

    def test_update_runs_completed(self):
        state = WorkspaceState.load_or_create(self.temp_dir.name)
        state.active_runs.append(
            {
                "run_id": "run_001",
                "modal_call_id": "call_123",
                "params": {"lr": 0.001},
                "started_at": "2026-01-01T00:00:00Z",
            }
        )

        with (
            patch(
                "hyppo.orchestrator.check_modal_run_status",
                return_value={"status": "completed"},
            ),
            patch(
                "hyppo.orchestrator.get_modal_run_result",
                return_value={"best_val_loss": 0.35},
            ),
            patch(
                "hyppo.orchestrator.fetch_run_metrics",
                return_value={
                    "epochs_completed": 10,
                    "best_val_loss": 0.35,
                    "best_epoch": 8,
                    "last_3_val_losses": [0.37, 0.36, 0.35],
                    "current_train_loss": 0.28,
                    "trend": "improving",
                },
            ),
        ):
            update_runs_from_modal_and_wandb(state)

        self.assertEqual(len(state.active_runs), 0)
        self.assertEqual(len(state.completed_runs), 1)
        self.assertEqual(state.completed_runs[0]["best_val_loss"], 0.35)
        self.assertEqual(state.completed_runs[0]["status"], "completed")

    def test_update_runs_keeps_unknown_modal_status_active(self):
        state = WorkspaceState.load_or_create(self.temp_dir.name)
        state.active_runs.append(
            {
                "run_id": "run_001",
                "modal_call_id": "call_123",
                "params": {"lr": 0.001},
                "started_at": "2026-01-01T00:00:00Z",
            }
        )

        with (
            patch(
                "hyppo.orchestrator.check_modal_run_status",
                return_value={"status": "unknown", "error": "temporary outage"},
            ),
            patch(
                "hyppo.orchestrator.fetch_run_metrics",
                return_value={
                    "epochs_completed": 3,
                    "best_val_loss": 0.41,
                    "best_epoch": 2,
                    "last_3_val_losses": [0.48, 0.44, 0.41],
                    "current_train_loss": 0.35,
                    "trend": "improving",
                },
            ),
        ):
            update_runs_from_modal_and_wandb(state)

        self.assertEqual(len(state.active_runs), 1)
        self.assertEqual(len(state.completed_runs), 0)
        self.assertEqual(state.active_runs[0]["last_error"]["source"], "modal_status")

    def test_wandb_history_normalization(self):
        from hyppo.tools.wandb_reader import _normalize_history

        rows = [{"epoch": 0, "val_loss": 0.5}, {"epoch": 1, "val_loss": 0.4}]
        self.assertEqual(_normalize_history(rows), rows)
