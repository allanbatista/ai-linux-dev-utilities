"""Unit tests for ab_cli.commands.auto_commit helpers."""
import json

from ab_cli.commands.auto_commit import (
    extract_json_object,
    normalize_branch_name,
)


class TestNormalizeBranchName:
    """Tests for branch name normalization."""

    def test_normalize_branch_name(self):
        """Normalizes whitespace and invalid characters."""
        result = normalize_branch_name("  Feature/Add User Auth!  ")
        assert result == "Feature/Add-User-Auth"

    def test_normalize_branch_name_truncates(self):
        """Truncates long branch names."""
        result = normalize_branch_name("feature/" + ("a" * 100))
        assert len(result) <= 50


class TestExtractJsonObject:
    """Tests for structured response parsing."""

    def test_extract_json_object_plain_json(self):
        """Parses plain JSON text."""
        payload = '{"branch_name": "feature/test", "commit_message": "feat: test"}'
        assert extract_json_object(payload)["branch_name"] == "feature/test"

    def test_extract_json_object_code_fence(self):
        """Parses JSON wrapped in code fences."""
        payload = "```json\n{\"branch_name\": \"feature/test\", \"commit_message\": \"feat: test\"}\n```"
        assert extract_json_object(payload)["commit_message"] == "feat: test"

    def test_extract_json_object_with_extra_text(self):
        """Parses JSON embedded in extra text."""
        payload = "Here is the result:\n{\"branch_name\": \"feature/test\", \"commit_message\": \"feat: test\"}\nDone."
        assert extract_json_object(payload)["branch_name"] == "feature/test"
