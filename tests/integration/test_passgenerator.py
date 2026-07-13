"""Integration tests for the password generator wrapper."""
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
AB = ROOT / "bin" / "ab"


def test_passgenerator_help_is_available():
    result = subprocess.run(
        [str(AB), "util", "passgenerator", "--help"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--min-digits" in result.stdout
