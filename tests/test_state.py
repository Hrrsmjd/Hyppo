import tempfile
import unittest

from hyppo.config import save_project_config
from hyppo.state import WorkspaceState


class WorkspaceStateTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        save_project_config(
            self.temp_dir.name,
            {
                "model_description": "Test model",
                "available_hyperparameters": ["learning_rate", "dropout"],
                "wandb_project": "test-project",
            },
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_load_or_create(self):
        state = WorkspaceState.load_or_create(self.temp_dir.name)
        self.assertEqual(state.active_runs, [])
        self.assertEqual(state.completed_runs, [])
        self.assertEqual(state.config["model_description"], "Test model")

    def test_search_space_lifecycle(self):
        state = WorkspaceState.load_or_create(self.temp_dir.name)
        self.assertFalse(state.search_space_exists())
        self.assertIsNone(state.search_space)

        space = {
            "version": 1,
            "parameters": {"lr": {"type": "continuous", "min": 1e-5, "max": 1e-2}},
            "changelog": [],
        }
        state.write_search_space(space)
        self.assertTrue(state.search_space_exists())
        self.assertEqual(state.read_search_space()["parameters"]["lr"]["min"], 1e-5)

    def test_active_runs_round_trip(self):
        state = WorkspaceState.load_or_create(self.temp_dir.name)
        state.active_runs.append({"run_id": "run_001", "params": {"lr": 0.001}})
        state.save_active_runs()

        reloaded = WorkspaceState.load_or_create(self.temp_dir.name)
        self.assertEqual(len(reloaded.active_runs), 1)
        self.assertEqual(reloaded.active_runs[0]["run_id"], "run_001")

    def test_best_completed_val_loss(self):
        state = WorkspaceState.load_or_create(self.temp_dir.name)
        state.completed_runs.extend(
            [
                {"run_id": "run_001", "best_val_loss": 0.5},
                {"run_id": "run_002", "best_val_loss": 0.3},
                {"run_id": "run_003", "best_val_loss": None, "status": "failed"},
            ]
        )
        self.assertEqual(state.best_completed_val_loss(), 0.3)

    def test_status_snapshot(self):
        state = WorkspaceState.load_or_create(self.temp_dir.name)
        state.completed_runs.append({"run_id": "run_001", "best_val_loss": 0.4})

        snapshot = state.status_snapshot()

        self.assertEqual(snapshot["active_runs"], 0)
        self.assertEqual(snapshot["completed_runs"], 1)
        self.assertEqual(snapshot["best_val_loss"], 0.4)
        self.assertIsNone(snapshot["search_space_version"])
