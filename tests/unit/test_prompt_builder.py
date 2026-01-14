#!/usr/bin/env python3
"""Unit tests for prompt_builder utilities."""
import pytest
from ab_cli.utils.prompt_builder import (
    build_generation_prompt,
    clean_llm_response,
    normalize_identifier,
    strip_markdown_code_block,
)


class TestBuildGenerationPrompt:
    """Tests for build_generation_prompt function."""

    def test_basic_prompt_structure(self):
        """Test that prompt has basic required structure."""
        result = build_generation_prompt(
            content="test content",
            rules=["Rule 1", "Rule 2"],
            lang="en",
            task_description="Test task",
        )
        assert "Test task" in result
        assert "RULES:" in result
        assert "1. Rule 1" in result
        assert "2. Rule 2" in result
        assert "CONTENT:" in result
        assert "test content" in result

    def test_language_added_when_not_in_rules(self):
        """Test that language is added when not mentioned in rules."""
        result = build_generation_prompt(
            content="test",
            rules=["Some rule"],
            lang="pt-br",
            task_description="Task",
        )
        assert "OUTPUT LANGUAGE: pt-br" in result

    def test_language_not_duplicated_if_in_rules(self):
        """Test that language is not duplicated if mentioned in rules."""
        result = build_generation_prompt(
            content="test",
            rules=["Write in language: en"],
            lang="en",
            task_description="Task",
        )
        # Should only appear once (in the rule)
        assert result.count("language") == 1

    def test_examples_section_added(self):
        """Test that examples are added when provided."""
        result = build_generation_prompt(
            content="test",
            rules=["Rule"],
            lang="en",
            task_description="Task",
            examples=["Example 1", "Example 2"],
        )
        assert "EXAMPLES:" in result
        assert "Example 1" in result
        assert "Example 2" in result

    def test_custom_response_instruction(self):
        """Test custom response instruction."""
        result = build_generation_prompt(
            content="test",
            rules=["Rule"],
            lang="en",
            task_description="Task",
            response_instruction="Return ONLY the branch name:",
        )
        assert "Return ONLY the branch name:" in result

    def test_empty_rules_list(self):
        """Test with empty rules list."""
        result = build_generation_prompt(
            content="test",
            rules=[],
            lang="en",
            task_description="Task",
        )
        assert "RULES:" not in result
        assert "CONTENT:" in result


class TestCleanLlmResponse:
    """Tests for clean_llm_response function."""

    def test_strips_whitespace(self):
        """Test that response is stripped of whitespace."""
        result = clean_llm_response("  hello world  ")
        assert result == "hello world"

    def test_strips_quotes_single(self):
        """Test stripping single quotes."""
        result = clean_llm_response("'feature/add-login'", strip_quotes=True)
        assert result == "feature/add-login"

    def test_strips_quotes_double(self):
        """Test stripping double quotes."""
        result = clean_llm_response('"feature/add-login"', strip_quotes=True)
        assert result == "feature/add-login"

    def test_strips_backticks(self):
        """Test stripping backticks."""
        result = clean_llm_response("`feature/add-login`", strip_quotes=True)
        assert result == "feature/add-login"

    def test_max_lines_single(self):
        """Test limiting to single line."""
        result = clean_llm_response("line1\nline2\nline3", max_lines=1)
        assert result == "line1"

    def test_max_lines_multiple(self):
        """Test limiting to multiple lines."""
        result = clean_llm_response("line1\nline2\nline3", max_lines=2)
        assert result == "line1\nline2"

    def test_max_lines_zero_means_unlimited(self):
        """Test that max_lines=0 means unlimited."""
        result = clean_llm_response("line1\nline2\nline3", max_lines=0)
        assert result == "line1\nline2\nline3"

    def test_max_length_truncates(self):
        """Test max length truncation."""
        result = clean_llm_response("hello world", max_length=5)
        assert result == "hello"

    def test_max_length_with_trim_char(self):
        """Test max length with trim character."""
        result = clean_llm_response("feature-add-login-", max_length=15, trim_char="-")
        assert result == "feature-add-log"
        assert not result.endswith("-")

    def test_empty_response(self):
        """Test with empty response."""
        result = clean_llm_response("")
        assert result == ""

    def test_none_response_handled(self):
        """Test handling of None-like falsy response."""
        result = clean_llm_response("")
        assert result == ""

    def test_strip_code_fences(self):
        """Test stripping markdown code fences."""
        result = clean_llm_response(
            "```python\nprint('hello')\n```",
            strip_code_fences=True
        )
        assert result == "print('hello')"

    def test_strip_code_fences_preserves_content(self):
        """Test that non-fenced content is preserved."""
        result = clean_llm_response("normal text", strip_code_fences=True)
        assert result == "normal text"


class TestNormalizeIdentifier:
    """Tests for normalize_identifier function."""

    def test_replaces_whitespace_with_separator(self):
        """Test whitespace replacement."""
        result = normalize_identifier("fix login bug")
        assert result == "fix-login-bug"

    def test_custom_separator(self):
        """Test custom separator."""
        result = normalize_identifier("fix login bug", separator="_")
        assert result == "fix_login_bug"

    def test_removes_invalid_characters(self):
        """Test removal of invalid characters."""
        result = normalize_identifier("fix: login (bug)")
        assert result == "fix-login-bug"

    def test_preserves_allowed_characters(self):
        """Test that allowed characters are preserved."""
        result = normalize_identifier("feature/add-user-123")
        assert result == "feature/add-user-123"

    def test_converts_to_lowercase(self):
        """Test lowercase conversion."""
        result = normalize_identifier("Feature/Add-User")
        assert result == "feature/add-user"

    def test_enforces_max_length(self):
        """Test max length enforcement."""
        result = normalize_identifier("very-long-branch-name-that-exceeds-limit", max_length=20)
        assert len(result) <= 20
        assert not result.endswith("-")

    def test_empty_input(self):
        """Test with empty input."""
        result = normalize_identifier("")
        assert result == ""

    def test_whitespace_only_input(self):
        """Test with whitespace only input."""
        result = normalize_identifier("   ")
        assert result == ""

    def test_multiple_consecutive_spaces(self):
        """Test multiple consecutive spaces."""
        result = normalize_identifier("fix   multiple   spaces")
        assert result == "fix-multiple-spaces"


class TestStripMarkdownCodeBlock:
    """Tests for strip_markdown_code_block function."""

    def test_strips_fenced_code_block(self):
        """Test stripping fenced code block."""
        result = strip_markdown_code_block("```python\nprint('hello')\n```")
        assert result == "print('hello')"

    def test_strips_fenced_code_block_no_language(self):
        """Test stripping fenced code block without language."""
        result = strip_markdown_code_block("```\ncode here\n```")
        assert result == "code here"

    def test_strips_inline_code(self):
        """Test stripping inline code."""
        result = strip_markdown_code_block("`inline code`")
        assert result == "inline code"

    def test_preserves_plain_text(self):
        """Test that plain text is preserved."""
        result = strip_markdown_code_block("plain text")
        assert result == "plain text"

    def test_empty_input(self):
        """Test with empty input."""
        result = strip_markdown_code_block("")
        assert result == ""

    def test_multiline_code_block(self):
        """Test multiline code block."""
        code = "```bash\necho 'line 1'\necho 'line 2'\necho 'line 3'\n```"
        result = strip_markdown_code_block(code)
        assert result == "echo 'line 1'\necho 'line 2'\necho 'line 3'"

    def test_code_block_with_leading_whitespace(self):
        """Test code block with leading whitespace."""
        result = strip_markdown_code_block("  ```python\ncode\n```")
        assert result == "code"


class TestUtilsModuleExports:
    """Tests for utils module exports."""

    def test_prompt_builder_functions_exported_from_utils(self):
        """Test that prompt builder functions are exported from utils."""
        from ab_cli.utils import (
            build_generation_prompt,
            clean_llm_response,
            normalize_identifier,
            strip_markdown_code_block,
        )
        assert callable(build_generation_prompt)
        assert callable(clean_llm_response)
        assert callable(normalize_identifier)
        assert callable(strip_markdown_code_block)
