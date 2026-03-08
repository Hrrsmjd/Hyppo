import io
import tempfile
import unittest
from contextlib import redirect_stdout

from hyppo.cli import handle_command
from hyppo.config import HyppoConfig, save_project_config


class CliTests(unittest.TestCase):
    def setUp(self):
        self.cfg = HyppoConfig()
        self.temp_dir = tempfile.TemporaryDirectory()
        save_project_config(
            self.temp_dir.name,
            {
                "model_description": "CLI test model",
                "available_hyperparameters": ["learning_rate"],
                "wandb_project": "test-project",
            },
        )
        self.cfg.project_dir = self.temp_dir.name

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_heartbeat_must_be_positive(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            handle_command("/heartbeat 0", self.cfg, None)

        self.assertEqual(self.cfg.heartbeat_minutes, 5)
        self.assertIn("Value must be positive", buffer.getvalue())

    def test_status_prints_snapshot(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            handle_command("/status", self.cfg, None)

        output = buffer.getvalue()
        self.assertIn("Active Runs:", output)
        self.assertIn("Completed Runs:", output)
