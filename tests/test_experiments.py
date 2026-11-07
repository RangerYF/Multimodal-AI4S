"""Contract checks for experiment isolation and auditable input fingerprints."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ai4s import experiments as runner


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.patch = patch.object(runner, "ROOT", self.root)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        (self.root / "input.txt").write_text("first")
        self.config = dict(schema_version=1, run_id="test-001", task="video",
                           args=["input.txt"], inputs=["input.txt"])

    def test_prevents_output_escape_and_override(self):
        for run_id in ["../escape", "/absolute", "a/b", ""]:
            with self.assertRaises(ValueError):
                runner.plan({**self.config, "run_id": run_id})
        for arg in ["--output", "--output=elsewhere"]:
            with self.assertRaises(ValueError):
                runner.plan({**self.config, "args": [arg]})

    def test_no_overwrite_even_empty_run(self):
        output, command = runner.plan(self.config)
        self.assertEqual(Path(command[-1]), output / "artifacts")
        self.assertFalse(output.exists())
        output.mkdir(parents=True)
        with self.assertRaises(FileExistsError):
            runner.plan(self.config)

    def test_input_content_change_is_detected(self):
        before = runner.fingerprint(self.config["inputs"])
        (self.root / "input.txt").write_text("other")
        after = runner.fingerprint(self.config["inputs"])
        self.assertNotEqual(before[0]["sha256"], after[0]["sha256"])

    def test_missing_input_fails_before_output_creation(self):
        with self.assertRaises(FileNotFoundError):
            runner.plan({**self.config, "inputs": ["missing"]})
        self.assertFalse((self.root / "outputs").exists())


if __name__ == "__main__":
    unittest.main()
