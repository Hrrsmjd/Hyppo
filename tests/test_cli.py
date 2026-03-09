import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from hyppo.cli import (
    CliSession,
    get_completion_candidates,
    handle_command,
    normalize_interactive_line,
)
from hyppo.config import HyppoConfig, save_project_config


class FakeThread:
    def __init__(self, alive: bool):
        self._alive = alive
        self.join_timeout = None

    def is_alive(self):
        return self._alive

    def join(self, timeout=None):
        self.join_timeout = timeout


class CliTests(unittest.TestCase):
    def setUp(self):
        self.cfg = HyppoConfig()
        self.root_dir = tempfile.TemporaryDirectory()
        self.project_dir = Path(self.root_dir.name) / "project"
        self.project_dir.mkdir()
        self.cwd_path = Path(self.root_dir.name) / "cwd"
        self.cwd_path.mkdir()
        self.external_dir = Path(self.root_dir.name) / "external"
        self.external_dir.mkdir()
        self.home_dir = Path(self.root_dir.name) / "home"
        self.home_dir.mkdir()

        project_path = self.project_dir
        (project_path / "train.py").write_text("print('train')\n", encoding="utf-8")
        (project_path / "params_config.py").write_text("x = 1\n", encoding="utf-8")
        (project_path / "params_dir").mkdir()
        (self.external_dir / "outside-alpha").mkdir()
        (self.external_dir / "outside-beta.py").write_text("print('x')\n", encoding="utf-8")
        (self.home_dir / "home-alpha").mkdir()
        save_project_config(
            self.project_dir,
            {
                "llm_description": "CLI test model",
                "available_hyperparameters": ["learning_rate"],
                "wandb_project": "test-project",
                "training_script": "train.py",
            },
        )
        (self.cwd_path / "project-alpha").mkdir()
        (self.cwd_path / "project-beta").mkdir()
        (self.cwd_path / "project.txt").write_text("not a dir\n", encoding="utf-8")

        self.cfg.project_dir = str(self.project_dir)
        self.cfg.script = "train.py"

    def tearDown(self):
        self.root_dir.cleanup()

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

        self.assertEqual(self.cfg.project_dir, str(self.project_dir))
        self.assertIn("Project directory is not writable", buffer.getvalue())

    def test_describe_appends_user_notes(self):
        handle_command("/describe first note", self.cfg, None)
        handle_command("/describe second note", self.cfg, None)

        self.assertIn("first note", self.cfg.user_description)
        self.assertIn("second note", self.cfg.user_description)

    def test_command_completion_matches_slash_prefix(self):
        candidates = get_completion_candidates("/pa", self.cfg, cwd=str(self.cwd_path))

        self.assertEqual([candidate.text for candidate in candidates], ["/params"])

    def test_project_path_completion_lists_directories_only(self):
        candidates = get_completion_candidates("/project @pro", self.cfg, cwd=str(self.cwd_path))

        self.assertEqual(
            [candidate.text for candidate in candidates],
            ["@project-alpha/", "@project-beta/"],
        )

    def test_script_path_completion_uses_project_root(self):
        candidates = get_completion_candidates("/script @pa", self.cfg, cwd=str(self.cwd_path))

        self.assertEqual(
            [candidate.text for candidate in candidates],
            ["@params_dir/", "@params_config.py"],
        )

    def test_normalize_interactive_line_strips_at_marker_for_path_commands(self):
        self.assertEqual(
            normalize_interactive_line("/project @project-alpha", self.cfg, cwd=str(self.cwd_path)),
            "/project project-alpha",
        )
        self.assertEqual(
            normalize_interactive_line("/script @train.py", self.cfg, cwd=str(self.cwd_path)),
            "/script train.py",
        )

    def test_path_completion_allows_parent_relative_paths(self):
        candidates = get_completion_candidates("/project @../ex", self.cfg, cwd=str(self.cwd_path))

        self.assertEqual([candidate.text for candidate in candidates], ["@../external/"])

    def test_path_completion_allows_absolute_paths(self):
        prefix = f"/project @{self.external_dir.parent}/ex"
        candidates = get_completion_candidates(prefix, self.cfg, cwd=str(self.cwd_path))

        self.assertEqual([candidate.text for candidate in candidates], [f"@{self.external_dir}/"])

    def test_path_completion_allows_home_paths(self):
        with patch.dict("os.environ", {"HOME": str(self.home_dir)}):
            candidates = get_completion_candidates("/project @~/ho", self.cfg, cwd=str(self.cwd_path))

        self.assertEqual([candidate.text for candidate in candidates], ["@~/home-alpha/"])

    def test_escape_stop_sets_event_for_running_campaign(self):
        session = CliSession(cfg=self.cfg, cwd=str(self.cwd_path))
        session.campaign_thread = FakeThread(alive=True)

        message = session.request_stop()

        self.assertEqual(message, "Stop signal sent. Campaign will stop after current heartbeat.")
        self.assertTrue(session.stop_event.is_set())

    def test_stop_command_ignores_stale_campaign_thread(self):
        session = CliSession(cfg=self.cfg, cwd=str(self.cwd_path))
        session.campaign_thread = FakeThread(alive=False)
        buffer = io.StringIO()

        with redirect_stdout(buffer):
            session.process_line("/stop")

        self.assertIn("No campaign running.", buffer.getvalue())

    def test_quit_joins_running_campaign_thread(self):
        session = CliSession(cfg=self.cfg, cwd=str(self.cwd_path))
        session.campaign_thread = FakeThread(alive=True)
        buffer = io.StringIO()

        with redirect_stdout(buffer):
            keep_running = session.process_line("/quit")

        self.assertFalse(keep_running)
        self.assertTrue(session.stop_event.is_set())
        self.assertEqual(session.campaign_thread.join_timeout, 5)
