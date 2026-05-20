"""File processing utilities for ab-cli.

This module handles file reading, binary detection,
directory traversal, .aiignore support, and token estimation.
"""
import os
import pathlib
import subprocess
from typing import List, Optional, Tuple

from binaryornot.check import is_binary
import pathspec

from ab_cli.utils.api import pp


def is_binary_file(file_path: pathlib.Path) -> bool:
    """
    Detect if a file is binary using the binaryornot library.

    Args:
        file_path: Path of the file to check.

    Returns:
        True if the file is binary, False if it's text.
    """
    try:
        return is_binary(str(file_path))
    except Exception:
        return True  # If can't read, assume binary


def find_git_root(start_path: pathlib.Path) -> Optional[pathlib.Path]:
    """
    Find the git repository root from the starting path.

    Args:
        start_path: Starting path for search.

    Returns:
        Git root path or None if not in a repository.
    """
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--show-toplevel'],
            capture_output=True, text=True, cwd=str(start_path)
        )
        if result.returncode == 0:
            return pathlib.Path(result.stdout.strip())
    except Exception:
        pass
    return None


def find_aiignore_files(start_path: pathlib.Path) -> List[pathlib.Path]:
    """
    Search for .aiignore files from starting directory to git root.

    Args:
        start_path: Starting path for search.

    Returns:
        List of .aiignore file paths found (from most specific to most general).
    """
    aiignore_files = []
    current = start_path.resolve()
    git_root = find_git_root(current)

    while current != current.parent:
        aiignore_path = current / '.aiignore'
        if aiignore_path.exists() and aiignore_path.is_file():
            aiignore_files.append(aiignore_path)

        # Stop at git root if found
        if git_root and current == git_root:
            break

        current = current.parent

    return aiignore_files


def load_aiignore_spec(aiignore_files: List[pathlib.Path]) -> Optional[pathspec.GitIgnoreSpec]:
    """
    Load and combine patterns from multiple .aiignore files.

    Args:
        aiignore_files: List of .aiignore paths (from most specific to most general).

    Returns:
        Combined spec or None if no patterns.
    """
    all_patterns = []

    # Process from most general (root) to most specific
    for aiignore_path in reversed(aiignore_files):
        try:
            with open(aiignore_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            all_patterns.extend(lines)
        except Exception as e:
            pp(f"Warning: Error reading {aiignore_path}: {e}")

    if not all_patterns:
        return None

    return pathspec.GitIgnoreSpec.from_lines(all_patterns)


def should_ignore_path(
    file_path: pathlib.Path,
    spec: Optional[pathspec.GitIgnoreSpec],
    base_path: pathlib.Path
) -> bool:
    """
    Check if a file should be ignored based on .aiignore patterns.

    Args:
        file_path: Absolute path of the file.
        spec: Compiled GitIgnore spec (or None).
        base_path: Base path for relative path calculation.

    Returns:
        True if the file should be ignored.
    """
    if spec is None:
        return False

    try:
        rel_path = file_path.relative_to(base_path)
        return spec.match_file(str(rel_path))
    except ValueError:
        # file_path is not relative to base_path
        return spec.match_file(str(file_path))


def process_file(file_path: pathlib.Path, path_format: str, max_tokens_doc: int) -> Tuple[str, int, int]:
    """
    Read file content, format header and truncate if necessary based on tokens.

    Args:
        file_path: Path of the file to process.
        path_format: How the path should be formatted ('full', 'relative', 'name_only').
        max_tokens_doc: Maximum estimated tokens for this file.

    Returns:
        Tuple containing formatted content, word count and estimated tokens.
    """
    try:
        display_path = ""
        if path_format == 'name_only':
            display_path = file_path.name
        elif path_format == 'relative':
            display_path = os.path.relpath(file_path.resolve(), pathlib.Path.cwd())
        else:  # 'full'
            display_path = str(file_path.resolve())

        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        original_tokens = len(content) // 4
        warning_message = ""

        if original_tokens > max_tokens_doc:
            max_chars = max_tokens_doc * 4
            content = content[:max_chars]
            warning_message = (
                f"// warning_content_truncated=\"true\" "
                f"original_token_count=\"{original_tokens}\" "
                f"new_token_count=\"{max_tokens_doc}\"\n"
            )
            pp(f"  -> Warning: File '{display_path}' was truncated to ~{max_tokens_doc} tokens.")

        word_count = len(content.split())
        estimated_tokens = len(content) // 4
        formatted_content = f"// filename=\"{display_path}\"\n{warning_message}{content}\n"

        return formatted_content, word_count, estimated_tokens
    except Exception as e:
        error_message = f"// error_processing_file=\"{file_path.resolve()}\"\n// Error: {e}\n"
        return error_message, 0, 0


def estimate_file_tokens(file_path: pathlib.Path) -> int:
    """
    Estimate the number of tokens in a file.

    Uses the approximation of ~4 characters per token.

    Args:
        file_path: Path to the file

    Returns:
        Estimated token count
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        return len(content) // 4
    except Exception:
        return 0


def get_directory_files(
    directory: pathlib.Path,
    aiignore_spec: Optional[pathspec.GitIgnoreSpec] = None
) -> List[pathlib.Path]:
    """
    Get all non-binary, non-ignored files from a directory recursively.

    Args:
        directory: Directory to scan
        aiignore_spec: Optional .aiignore spec for filtering

    Returns:
        List of file paths to process
    """
    files = []
    base_path = directory.resolve()

    for child_path in directory.rglob('*'):
        if not child_path.is_file():
            continue

        # Check .aiignore
        if should_ignore_path(child_path.resolve(), aiignore_spec, base_path):
            continue

        # Check if binary
        if is_binary_file(child_path):
            continue

        files.append(child_path)

    return files
