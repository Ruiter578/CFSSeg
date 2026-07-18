import os
import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "tools" / "run_pseudo_label_matched_global_20260718.sh"


class MatchedGlobalRunnerTests(unittest.TestCase):
    def run_runner(self, dry_run):
        env = os.environ.copy()
        env["DRY_RUN"] = dry_run
        return subprocess.run(
            ["bash", str(RUNNER)],
            cwd=REPO_ROOT,
            env=env,
            check=False,
            text=True,
            capture_output=True,
        )

    def test_rejects_invalid_dry_run(self):
        result = self.run_runner("invalid")

        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("DRY_RUN must be 0 or 1", result.stderr)

    def test_refuses_dirty_worktree(self):
        marker = REPO_ROOT / ".matched-global-runner-dirty-test"
        marker.write_text("force dirty status\n", encoding="utf-8")
        try:
            result = self.run_runner("1")
        finally:
            marker.unlink(missing_ok=True)

        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("refusing dirty worktree", result.stderr)


if __name__ == "__main__":
    unittest.main()
