#!/usr/bin/env python3
"""
Prompt builder utilities for LLM generation tasks.

Provides reusable functions for building prompts, cleaning LLM responses,
and normalizing identifiers (branch names, etc.).
"""
import re
from typing import List, Optional


def build_generation_prompt(
    content: str,
    rules: List[str],
    lang: str,
    task_description: str,
    examples: Optional[List[str]] = None,
    response_instruction: str = "Respond ONLY with the result:"
) -> str:
    """
    Build a structured prompt for LLM generation tasks.

    Creates a consistent prompt format with task description, rules,
    content, and response instructions.

    Args:
        content: The main content to analyze (diff, code, description, etc.)
        rules: List of rules/instructions for the LLM to follow
        lang: Output language code (e.g., 'en', 'pt-br')
        task_description: Description of what the LLM should do
        examples: Optional list of example outputs
        response_instruction: Final instruction line (default: "Respond ONLY with the result:")

    Returns:
        Formatted prompt string ready to send to LLM.

    Example:
        >>> prompt = build_generation_prompt(
        ...     content="fix login button",
        ...     rules=["Use kebab-case", "Max 50 chars"],
        ...     lang="en",
        ...     task_description="Generate a git branch name"
        ... )
    """
    prompt_parts = [task_description, ""]

    # Add rules section
    if rules:
        prompt_parts.append("RULES:")
        for i, rule in enumerate(rules, 1):
            prompt_parts.append(f"{i}. {rule}")
        prompt_parts.append("")

    # Add language if not already in rules
    lang_in_rules = any("language" in rule.lower() for rule in rules)
    if not lang_in_rules and lang:
        prompt_parts.append(f"OUTPUT LANGUAGE: {lang}")
        prompt_parts.append("")

    # Add content
    prompt_parts.append("CONTENT:")
    prompt_parts.append(content)
    prompt_parts.append("")

    # Add examples if provided
    if examples:
        prompt_parts.append("EXAMPLES:")
        for example in examples:
            prompt_parts.append(f"  {example}")
        prompt_parts.append("")

    # Add response instruction
    prompt_parts.append(response_instruction)

    return "\n".join(prompt_parts)


def clean_llm_response(
    response: str,
    max_lines: int = 0,
    strip_quotes: bool = True,
    strip_code_fences: bool = False,
    max_length: Optional[int] = None,
    trim_char: str = ""
) -> str:
    """
    Standardized cleanup for LLM responses.

    Applies common cleanup operations to raw LLM output.

    Args:
        response: Raw LLM response text
        max_lines: Maximum number of lines to keep (0 = unlimited)
        strip_quotes: Remove surrounding quotes (", ', `)
        strip_code_fences: Remove markdown code fences (```)
        max_length: Maximum character length (None = unlimited)
        trim_char: Character to trim from end when truncating (e.g., '-')

    Returns:
        Cleaned response string.

    Example:
        >>> clean_llm_response('"feature/add-login"', strip_quotes=True)
        'feature/add-login'
        >>> clean_llm_response("line1\\nline2\\nline3", max_lines=1)
        'line1'
    """
    if not response:
        return ""

    result = response.strip()

    # Strip markdown code fences
    if strip_code_fences and result.startswith('```'):
        lines = result.split('\n')
        # Remove first line (```language)
        lines = lines[1:]
        # Remove last line if it's closing fence
        if lines and lines[-1].strip() == '```':
            lines = lines[:-1]
        result = '\n'.join(lines)

    # Strip surrounding quotes
    if strip_quotes:
        result = result.strip('"\'`')

    # Limit to max_lines
    if max_lines > 0:
        lines = result.split('\n')
        result = '\n'.join(lines[:max_lines]).strip()

    # Enforce max length
    if max_length is not None and len(result) > max_length:
        result = result[:max_length]
        if trim_char:
            result = result.rstrip(trim_char)

    return result


def normalize_identifier(
    text: str,
    separator: str = "-",
    max_length: int = 50,
    allowed_chars: str = r'a-zA-Z0-9/_-',
    preserve_prefix_separator: bool = True
) -> str:
    """
    Normalize text for use as branch names, identifiers, etc.

    Converts text to a valid identifier by replacing spaces,
    removing invalid characters, and enforcing length limits.

    Args:
        text: Input text to normalize
        separator: Character to replace whitespace with (default: '-')
        max_length: Maximum length of output (default: 50)
        allowed_chars: Regex character class of allowed characters
        preserve_prefix_separator: If True, keeps '/' for prefixes like 'feature/'

    Returns:
        Normalized identifier string.

    Example:
        >>> normalize_identifier("Fix Login Bug")
        'fix-login-bug'
        >>> normalize_identifier("feature/Add User Auth")
        'feature/add-user-auth'
        >>> normalize_identifier("Very Long Name " * 10, max_length=20)
        'very-long-name-very'
    """
    if not text:
        return ""

    result = text.strip()

    # Replace whitespace with separator
    result = re.sub(r'\s+', separator, result)

    # Remove invalid characters (keep allowed_chars)
    pattern = f'[^{allowed_chars}]'
    result = re.sub(pattern, '', result)

    # Convert to lowercase for consistency
    result = result.lower()

    # Enforce max length
    if len(result) > max_length:
        result = result[:max_length].rstrip(separator)

    return result


def strip_markdown_code_block(text: str) -> str:
    """
    Remove markdown code block wrappers from text.

    Handles both fenced code blocks (```...```) and inline code (`...`).

    Args:
        text: Text potentially wrapped in markdown code block

    Returns:
        Text with code block markers removed.

    Example:
        >>> strip_markdown_code_block('```python\\nprint("hello")\\n```')
        'print("hello")'
    """
    if not text:
        return ""

    result = text.strip()

    # Handle fenced code blocks
    if result.startswith('```'):
        lines = result.split('\n')
        # Remove opening fence (with optional language)
        lines = lines[1:]
        # Remove closing fence
        if lines and lines[-1].strip() == '```':
            lines = lines[:-1]
        result = '\n'.join(lines)

    # Handle inline code (single backticks)
    elif result.startswith('`') and result.endswith('`') and result.count('`') == 2:
        result = result[1:-1]

    return result
