"""Unit tests for ab_cli.utils.git_helpers module."""
import subprocess

import pytest

from ab_cli.utils.git_helpers import get_commit_diff, run_git


class TestRunGit:
    """Tests for run_git function."""

    def test_run_git_handles_non_utf8_output(self, mock_git_repo, monkeypatch):
        """run_git handles non-UTF-8 bytes in git output without raising error."""
        monkeypatch.chdir(mock_git_repo)

        # Create a file with Latin-1 encoded content (non-UTF-8)
        # 0xe9 is 'é' in Latin-1, but invalid as a standalone byte in UTF-8
        latin1_file = mock_git_repo / "latin1.txt"
        latin1_file.write_bytes(b"Caf\xe9 is coffee in French\n")

        subprocess.run(["git", "add", "."], cwd=mock_git_repo, check=True)
        subprocess.run(
            ["git", "commit", "-m", "Add Latin-1 file"],
            cwd=mock_git_repo,
            check=True
        )

        # This should not raise UnicodeDecodeError
        diff = get_commit_diff("HEAD")

        # The output should contain the filename and replacement character
        assert "latin1.txt" in diff
        # Either contains Caf with replacement char or original partial content
        assert "Caf" in diff

    def test_run_git_handles_binary_content(self, mock_git_repo, monkeypatch):
        """run_git handles binary content in git output."""
        monkeypatch.chdir(mock_git_repo)

        # Create a file with various invalid UTF-8 byte sequences
        binary_file = mock_git_repo / "binary.bin"
        binary_file.write_bytes(b"\x80\x81\x82\xff\xfe binary content")

        subprocess.run(["git", "add", "."], cwd=mock_git_repo, check=True)
        subprocess.run(
            ["git", "commit", "-m", "Add binary file"],
            cwd=mock_git_repo,
            check=True
        )

        # This should not raise UnicodeDecodeError
        diff = get_commit_diff("HEAD")

        # The diff should be returned (may show binary file notice or content)
        assert isinstance(diff, str)

    def test_run_git_preserves_valid_utf8(self, mock_git_repo, monkeypatch):
        """run_git preserves valid UTF-8 content correctly."""
        monkeypatch.chdir(mock_git_repo)

        # Create a file with valid UTF-8 content including non-ASCII
        utf8_file = mock_git_repo / "utf8.txt"
        utf8_file.write_text("Café résumé naïve\n", encoding="utf-8")

        subprocess.run(["git", "add", "."], cwd=mock_git_repo, check=True)
        subprocess.run(
            ["git", "commit", "-m", "Add UTF-8 file"],
            cwd=mock_git_repo,
            check=True
        )

        diff = get_commit_diff("HEAD")

        # Valid UTF-8 should be preserved
        assert "Café" in diff
        assert "résumé" in diff
        assert "naïve" in diff
