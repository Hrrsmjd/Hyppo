import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from hyppo.cli import handle_command
from hyppo.config import HyppoConfig, save_project_config


class CliTests(unittest.TestCase):
    def setUp(self):
        self.cfg = HyppoConfig()
        self.temp_dir = tempfile.TemporaryDirectory()
        project_path = Path(self.temp_dir.name)
        (project_path / "train.py").write_text("print('train')\n", encoding="utf-8")
        save_project_config(
            self.temp_dir.name,
            {
                "llm_description": "CLI test model",
                "available_hyperparameters": ["learning_rate"],
                "wandb_project": "test-project",
                "training_script": "train.py",
            },
        )
        self.cfg.project_dir = self.temp_dir.name
        self.cfg.script = "train.py"

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
        self.assertIn("Runs Remaining:", output)

    def test_project_must_be_writable(self):
        buffer = io.StringIO()
        with (
            patch("hyppo.cli.os.path.isdir", return_value=True),
            patch("hyppo.cli.is_project_dir_writable", return_value=False),
            redirect_stdout(buffer),
        ):
            handle_command("/project /", self.cfg, None)

        self.assertEqual(self.cfg.project_dir, self.temp_dir.name)
        self.assertIn("Project directory is not writable", buffer.getvalue())

    def test_describe_appends_user_notes(self):
        handle_command("/describe first note", self.cfg, None)
        handle_command("/describe second note", self.cfg, None)

        self.assertIn("first note", self.cfg.user_description)
        self.assertIn("second note", self.cfg.user_description)
