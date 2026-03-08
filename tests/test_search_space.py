import tempfile
import unittest

from hyppo.config import save_project_config
from hyppo.state import WorkspaceState
from hyppo.tools.search_space import (
    execute_initialize_search_space,
    execute_update_search_space,
)


class SearchSpaceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        save_project_config(
            self.temp_dir.name,
            {
                "llm_description": "Test model",
                "available_hyperparameters": [
                    "learning_rate",
                    "dropout",
                    "weight_decay",
                ],
                "wandb_project": "test-project",
            },
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_initialize_search_space(self):
        state = WorkspaceState.load_or_create(self.temp_dir.name)
        result = execute_initialize_search_space(
            {
                "learning_rate": {
                    "type": "continuous",
                    "min": 1e-5,
                    "max": 1e-2,
                    "scale": "log",
                    "notes": "Typical range",
                }
            },
            state,
        )
        self.assertEqual(result["status"], "created")
        self.assertEqual(state.read_search_space()["version"], 1)

    def test_update_search_space(self):
        state = WorkspaceState.load_or_create(self.temp_dir.name)
        execute_initialize_search_space(
            {
                "learning_rate": {
                    "type": "continuous",
                    "min": 1e-5,
                    "max": 1e-2,
                    "scale": "log",
                    "notes": "Initial range",
                }
            },
            state,
        )

        result = execute_update_search_space(
            {"learning_rate": {"min": 1e-4, "max": 5e-3, "notes": "Narrowed"}},
            "Narrowed learning rate",
            state,
        )
        self.assertEqual(result["status"], "updated")
        self.assertEqual(
            state.read_search_space()["parameters"]["learning_rate"]["min"],
            1e-4,
        )
