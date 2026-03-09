"""Full heartbeat loop with mocked LLM (OpenAI format)."""

import json
import sys
import tempfile
import unittest
from types import SimpleNamespace
from dataclasses import dataclass, field
from unittest.mock import patch

from hyppo.config import save_project_config
from hyppo.orchestrator import (
    backfill_completed_run_metrics,
    run_heartbeat,
    update_runs_from_modal_and_wandb,
)
from hyppo.prompt_builder import build_prompt, format_state_for_prompt
from hyppo.state import WorkspaceState
from hyppo.tools.modal_runner import execute_launch_run, launch_modal_run


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
                "llm_description": "Test transformer for sentiment analysis.",
                "user_description": "Prefer aggressive search around learning rate.",
                "available_hyperparameters": [
                    "learning_rate",
                    "dropout",
                    "num_layers",
                ],
                "wandb_project": "test-project",
                "llm_provider": "anthropic",
                "llm_model": "claude-sonnet-4-20250514",
                "max_total_runs": 3,
                "max_concurrent_runs": 2,
                "max_time": 30,
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
                    "best_time_seconds": 100 + index,
                    "params": {"lr": 0.001},
                    "status": "completed",
                    "metric_history": [],
                }
            )

        text = format_state_for_prompt(state)
        self.assertIn("Older Completed Runs", text)
        self.assertIn("Recent Completed Runs", text)
        self.assertNotIn("Metric History", text)

    def test_active_run_history_is_truncated_in_prompt(self):
        state = WorkspaceState.load_or_create(self.temp_dir.name)
        state.active_runs.append(
            {
                "run_id": "run_001",
                "status": "running",
                "elapsed_time_seconds": 120.0,
                "progress_percent": 20.0,
                "best_val_loss": 0.5,
                "best_time_seconds": 120.0,
                "trend": "improving",
                "params": {"learning_rate": 0.001},
                "metric_history": [
                    {
                        "time_seconds": float(index),
                        "progress_percent": float(index),
                        "val_loss": 1.0 - index * 0.01,
                        "train_loss": 1.2 - index * 0.01,
                    }
                    for index in range(8)
                ],
            }
        )

        text = format_state_for_prompt(state)
        self.assertIn("Showing the most recent 5 of 8 history points.", text)

    def test_historical_insights_not_included_in_prompt(self):
        state = WorkspaceState.load_or_create(self.temp_dir.name)
        state.write_strategy("Current strategy")
        state.write_strategy("Updated strategy")

        text = format_state_for_prompt(state)
        self.assertIn("## Strategy", text)
        self.assertNotIn("## Historical Insights", text)

    def test_prompt_includes_description_tags(self):
        state = WorkspaceState.load_or_create(self.temp_dir.name)
        prompt = build_prompt(state)
        self.assertIn("<llm_description>", prompt)
        self.assertIn("<user_description>", prompt)

    def test_first_heartbeat_instructions_present(self):
        state = WorkspaceState.load_or_create(self.temp_dir.name)
        prompt = build_prompt(state)
        self.assertIn("No search space has been defined yet", prompt)

    def test_prompt_requires_strategy_before_launching_runs(self):
        state = WorkspaceState.load_or_create(self.temp_dir.name)
        prompt = build_prompt(state)
        self.assertIn("Before launching any new runs on a heartbeat", prompt)
        self.assertIn("Preferred order when launching is", prompt)

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

    def test_run_heartbeat_survives_missing_changelog_entry(self):
        state = WorkspaceState.load_or_create(self.temp_dir.name)
        client = FakeClient(
            responses=[
                make_tool_response(
                    [
                        {
                            "id": "tool_1",
                            "name": "update_strategy",
                            "input": {"content": "Insight: Narrow learning rate next."},
                        },
                        {
                            "id": "tool_2",
                            "name": "update_search_space",
                            "input": {
                                "updates": {
                                    "learning_rate": {
                                        "type": "continuous",
                                        "min": 1e-4,
                                        "max": 1e-2,
                                        "scale": "log",
                                        "notes": "Narrowed after early results.",
                                    }
                                }
                            },
                        },
                    ]
                ),
                make_text_response("Recovered after tool validation error."),
            ]
        )

        run_heartbeat(state, client=client)

        self.assertEqual(state.strategy, "Insight: Narrow learning rate next.")
        self.assertFalse(state.search_space_exists())

    def test_run_heartbeat_survives_missing_updates(self):
        state = WorkspaceState.load_or_create(self.temp_dir.name)
        client = FakeClient(
            responses=[
                make_tool_response(
                    [
                        {
                            "id": "tool_1",
                            "name": "update_strategy",
                            "input": {"content": "Insight: Search space change needs correction."},
                        },
                        {
                            "id": "tool_2",
                            "name": "update_search_space",
                            "input": {"changelog_entry": "Attempted to refine around the best run."},
                        },
                    ]
                ),
                make_text_response("Recovered after missing updates error."),
            ]
        )

        run_heartbeat(state, client=client)

        self.assertEqual(
            state.strategy,
            "Insight: Search space change needs correction.",
        )
        self.assertFalse(state.search_space_exists())

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

    def test_launch_run_respects_total_budget(self):
        state = WorkspaceState.load_or_create(self.temp_dir.name)
        state.completed_runs.extend(
            [
                {"run_id": "run_001", "best_val_loss": 0.4},
                {"run_id": "run_002", "best_val_loss": 0.3},
                {"run_id": "run_003", "best_val_loss": 0.2},
            ]
        )

        result = execute_launch_run({"learning_rate": 0.001}, state)

        self.assertEqual(result["error"], "Max total runs reached")

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

    def test_launch_modal_run_passes_max_time_minutes_when_supported(self):
        calls = []

        class FakeFunction:
            def spawn(self, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(object_id="call_123")

        fake_modal = SimpleNamespace(
            Function=SimpleNamespace(from_name=lambda app_name, function_name: FakeFunction())
        )

        with patch.dict(sys.modules, {"modal": fake_modal}):
            result = launch_modal_run(
                "run_001",
                {"learning_rate": 0.001},
                {
                    "modal_app_name": "hpo-agent",
                    "modal_function_name": "train_model",
                    "wandb_project": "test-project",
                    "wandb_entity": None,
                    "max_time": 15,
                },
            )

        self.assertEqual(result["modal_call_id"], "call_123")
        self.assertEqual(calls[0]["max_time_minutes"], 15)

    def test_launch_modal_run_retries_without_max_time_minutes_if_unsupported(self):
        calls = []

        class FakeFunction:
            def spawn(self, **kwargs):
                calls.append(kwargs)
                if "max_time_minutes" in kwargs:
                    raise TypeError("train_model() got an unexpected keyword argument 'max_time_minutes'")
                return SimpleNamespace(object_id="call_456")

        fake_modal = SimpleNamespace(
            Function=SimpleNamespace(from_name=lambda app_name, function_name: FakeFunction())
        )

        with patch.dict(sys.modules, {"modal": fake_modal}):
            result = launch_modal_run(
                "run_001",
                {"learning_rate": 0.001},
                {
                    "modal_app_name": "hpo-agent",
                    "modal_function_name": "train_model",
                    "wandb_project": "test-project",
                    "wandb_entity": None,
                    "max_time": 15,
                },
            )

        self.assertEqual(result["modal_call_id"], "call_456")
        self.assertEqual(len(calls), 2)
        self.assertIn("max_time_minutes", calls[0])
        self.assertNotIn("max_time_minutes", calls[1])

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
                    "metric_history": [
                        {
                            "time_seconds": 60.0,
                            "progress_percent": 10.0,
                            "val_loss": 0.37,
                            "train_loss": 0.29,
                        },
                        {
                            "time_seconds": 120.0,
                            "progress_percent": 20.0,
                            "val_loss": 0.35,
                            "train_loss": 0.28,
                        },
                    ],
                    "history_points": 2,
                    "best_val_loss": 0.35,
                    "best_time_seconds": 120.0,
                    "best_progress_percent": 20.0,
                    "latest_val_loss": 0.35,
                    "latest_train_loss": 0.28,
                    "elapsed_time_seconds": 120.0,
                    "progress_percent": 20.0,
                    "trend": "improving",
                },
            ),
        ):
            update_runs_from_modal_and_wandb(state)

        self.assertEqual(len(state.active_runs), 0)
        self.assertEqual(len(state.completed_runs), 1)
        self.assertEqual(state.completed_runs[0]["best_val_loss"], 0.35)
        self.assertEqual(state.completed_runs[0]["status"], "completed")

    def test_update_runs_completed_keeps_modal_metrics_when_wandb_empty(self):
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
                return_value={
                    "best_val_loss": 0.35,
                    "best_time_seconds": 120.0,
                    "best_progress_percent": 20.0,
                    "final_val_loss": 0.37,
                },
            ),
            patch(
                "hyppo.orchestrator.fetch_run_metrics",
                return_value={
                    "metric_history": [],
                    "history_points": 0,
                    "best_val_loss": None,
                    "best_time_seconds": None,
                    "best_progress_percent": None,
                    "latest_val_loss": None,
                    "latest_train_loss": None,
                    "elapsed_time_seconds": None,
                    "progress_percent": None,
                    "trend": "insufficient_data",
                },
            ),
        ):
            update_runs_from_modal_and_wandb(state)

        completed = state.completed_runs[0]
        self.assertEqual(completed["best_val_loss"], 0.35)
        self.assertEqual(completed["latest_val_loss"], 0.37)
        self.assertEqual(completed["history_points"], 1)
        self.assertEqual(completed["metric_history"][0]["val_loss"], 0.37)

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
                    "metric_history": [
                        {
                            "time_seconds": 180.0,
                            "progress_percent": 30.0,
                            "val_loss": 0.41,
                            "train_loss": 0.35,
                        }
                    ],
                    "history_points": 1,
                    "best_val_loss": 0.41,
                    "best_time_seconds": 180.0,
                    "best_progress_percent": 30.0,
                    "latest_val_loss": 0.41,
                    "latest_train_loss": 0.35,
                    "elapsed_time_seconds": 180.0,
                    "progress_percent": 30.0,
                    "trend": "improving",
                },
            ),
        ):
            update_runs_from_modal_and_wandb(state)

        self.assertEqual(len(state.active_runs), 1)
        self.assertEqual(len(state.completed_runs), 0)
        self.assertEqual(state.active_runs[0]["last_error"]["source"], "modal_status")

    def test_backfill_completed_run_metrics(self):
        state = WorkspaceState.load_or_create(self.temp_dir.name)
        state.completed_runs.append(
            {
                "run_id": "run_001",
                "status": "completed",
                "metric_history": [],
                "history_points": 0,
                "best_val_loss": None,
                "latest_val_loss": None,
            }
        )

        with patch(
            "hyppo.orchestrator.fetch_run_metrics",
            return_value={
                "metric_history": [
                    {
                        "time_seconds": 30.0,
                        "progress_percent": 5.0,
                        "val_loss": 0.5,
                        "train_loss": 0.6,
                    }
                ],
                "history_points": 1,
                "best_val_loss": 0.5,
                "best_time_seconds": 30.0,
                "best_progress_percent": 5.0,
                "latest_val_loss": 0.5,
                "latest_train_loss": 0.6,
                "elapsed_time_seconds": 30.0,
                "progress_percent": 5.0,
                "trend": "insufficient_data",
            },
        ):
            backfill_completed_run_metrics(state)

        completed = state.completed_runs[0]
        self.assertEqual(completed["history_points"], 1)
        self.assertEqual(completed["best_val_loss"], 0.5)

    def test_wandb_history_normalization(self):
        from hyppo.tools.wandb_reader import _normalize_history

        rows = [{"_runtime": 0, "val_loss": 0.5}, {"_runtime": 30, "val_loss": 0.4}]
        self.assertEqual(_normalize_history(rows), rows)

    def test_wandb_reader_prefers_scan_history(self):
        from hyppo.tools.wandb_reader import _read_run_history

        class FakeRun:
            def scan_history(self):
                return [{"val_loss": 0.5}]

        self.assertEqual(_read_run_history(FakeRun()), [{"val_loss": 0.5}])
