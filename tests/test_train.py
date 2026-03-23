"""Smoke test for training entry point."""

import os
import sys
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestTraining:
    def test_short_training_runs(self):
        """A 50-step training run should complete and print val_bpb."""
        result = subprocess.run(
            [sys.executable, "train.py", "--max-steps", "50",
             "--name", "test_smoke", "--no-nativebit"],
            capture_output=True, text=True, timeout=120,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        assert result.returncode == 0, f"Failed: {result.stderr[-500:]}"
        assert "val_bpb:" in result.stdout
