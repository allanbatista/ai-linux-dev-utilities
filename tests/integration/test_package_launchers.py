"""Integration tests for packaged CLI launchers."""
import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
AB = ROOT / "bin" / "ab"


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (["config", "path"], ["-m", "ab_cli.commands.config_cli", "path"]),
        (["models", "list"], ["-m", "ab_cli.commands.models", "list"]),
        (["prompt", "-p", "test"], ["-m", "ab_cli.commands.prompt", "-p", "test"]),
        (["git", "branch-name", "test"], ["-m", "ab_cli.commands.branch_name", "test"]),
        (["util", "explain", "test"], ["-m", "ab_cli.commands.explain", "test"]),
        (["media", "transcribe", "test"], ["-m", "ab_cli.commands.media", "transcribe", "test"]),
    ],
)
def test_launchers_use_packaged_python(tmp_path, args, expected):
    python = tmp_path / "python"
    python.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@"\n', encoding="utf-8")
    python.chmod(0o755)

    result = subprocess.run(
        [str(AB), *args],
        capture_output=True,
        env={**os.environ, "AB_PYTHON": str(python)},
        text=True,
    )

    assert result.returncode == 0
    assert result.stdout.splitlines() == expected
