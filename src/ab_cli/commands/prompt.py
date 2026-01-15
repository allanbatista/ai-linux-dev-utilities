#!/usr/bin/env python3
"""
CLI to concatenate file contents and send to OpenRouter.

Configuration via `~/.ab/config.json` (optional).

Example `~/.ab/config.json`:
{
  "global": {
    "language": "en",
    "api_base": "https://openrouter.ai/api/v1",
    "api_key_env": "OPENROUTER_API_KEY",
    "timeout_seconds": 300
  },
  "models": {
    "default": "nvidia/nemotron-3-nano-30b-a3b:free"
  }
}

Flag `--set-default-model <model>` to **persist** the default model.
"""
import argparse
import json
import pathlib
import sys
from typing import Any, Dict

import pyperclip

from ab_cli.core.config import get_config
from ab_cli.utils.error_handling import handle_cli_errors
from ab_cli.utils.api import (
    send_to_openrouter,
    build_specialist_prefix,
    set_verbose as set_api_verbose,
    pp,
)
from ab_cli.utils.file_processing import (
    is_binary_file,
    find_aiignore_files,
    load_aiignore_spec,
    should_ignore_path,
    process_file,
)
from ab_cli.utils.history import (
    sanitize_sensitive_data,
    save_to_history,
)

# Re-export for backward compatibility (tests import these from prompt.py)
__all__ = [
    'main',
    'send_to_openrouter',
    'sanitize_sensitive_data',
    'save_to_history',
]

VERBOSE = True


def _sync_verbose():
    """Sync the VERBOSE flag with the api module."""
    set_api_verbose(VERBOSE)


# =========================
# Utilities and Persistence
# =========================

def load_config() -> Dict[str, Any]:
    """Load config from ~/.ab/config.json using centralized config module."""
    try:
        config = get_config()
        # Return in legacy format for compatibility
        return {
            "model": config.get("models.default"),
            "api_base": config.get("global.api_base"),
            "api_key_env": config.get("global.api_key_env"),
            "request": {
                "timeout_seconds": config.get("global.timeout_seconds", 300)
            }
        }
    except Exception as e:
        pp(f"Warning: could not read config: {e}")
    return {}


def persist_default_model(new_model: str) -> bool:
    """
    Update default model in ~/.ab/config.json (models.default key),
    preserving other fields. Creates the file if it doesn't exist.
    """
    try:
        config = get_config()
        config.set("models.default", new_model)
        return True
    except Exception as e:
        pp(f"Error persisting default model: {e}")
        return False


# =========================
# Effective Configuration
# =========================

def resolve_settings(args, config: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve model/timeout/api_base/api_key_env from args + config."""
    # Model precedence: CLI > config > default
    model = args.model or config.get("model") or "nvidia/nemotron-3-nano-30b-a3b:free"

    # API key env var name
    api_key_env = config.get("api_key_env") or "OPENROUTER_API_KEY"

    # API base
    api_base = config.get("api_base") or "https://openrouter.ai/api/v1"

    # Timeout
    timeout_s = int(config.get("request", {}).get("timeout_seconds", 300))

    return {
        "model": model,
        "api_key_env": api_key_env,
        "api_base": api_base,
        "timeout_s": timeout_s,
    }


# =========================
# Main
# =========================


@handle_cli_errors
def main():
    """Main function that orchestrates script execution."""
    parser = argparse.ArgumentParser(
        description=(
            "Concatenate text file contents (ignores binaries) and "
            "optionally send to OpenRouter API.\n"
            "Use .aiignore to exclude files (gitignore syntax)."
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "paths",
        metavar="PATH",
        type=pathlib.Path,
        nargs='*',
        help="A list of files and/or directories to process."
    )
    parser.add_argument(
        "-p", "--prompt",
        type=str,
        help="An optional prompt to send to the API. Use '-' to read from stdin."
    )
    parser.add_argument(
        '--lang',
        type=str,
        default='en',
        help='Output language. Default: en'
    )
    parser.add_argument(
        '-n', '--max-tokens',
        type=int,
        default=900_000,
        help='Maximum estimated tokens for total context. Default: 900000'
    )
    parser.add_argument(
        '-nn', '--max-tokens-doc',
        type=int,
        default=250_000,
        help='Maximum estimated tokens per individual file. Default: 250000'
    )
    parser.add_argument(
        '-s', '--specialist',
        type=str,
        choices=['dev', 'rm'],
        help=(
            "Define a specialist persona:\n"
            "'dev' for Senior Programmer\n"
            "'rm'  for Senior Retail Media Analyst."
        )
    )
    parser.add_argument(
        '--model',
        type=str,
        help='OpenRouter model name to use. Ex: nvidia/nemotron-3-nano-30b-a3b:free'
    )
    parser.add_argument(
        '-m', '--max-completion-tokens',
        type=int,
        default=16000,
        help='Maximum tokens for model response. Default: 16000'
    )
    parser.add_argument(
        '-u', '--unlimited',
        action='store_true',
        help='Remove response token limit (does not send max_tokens to API)'
    )
    parser.add_argument(
        '--set-default-model',
        type=str,
        help='Set and persist the default model (top-level "model") in ~/.ab/config.json and exit.'
    )
    parser.add_argument(
        '--only-output',
        action='store_true',
        help="Return only the model result"
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help="Format result as JSON"
    )

    path_options = parser.add_mutually_exclusive_group()
    path_options.add_argument(
        "--relative-paths",
        action="store_true",
        help="Display relative paths instead of absolute paths."
    )
    path_options.add_argument(
        "--filename-only",
        action="store_true",
        help="Display only the filename instead of full path."
    )


    # If no arguments passed, show help
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    # If prompt is '-', read from stdin
    if args.prompt == '-':
        args.prompt = sys.stdin.read()

    global VERBOSE
    VERBOSE = not args.only_output
    _sync_verbose()

    # Update default model if requested
    if args.set_default_model:
        if persist_default_model(args.set_default_model):
            pp(f"Default model updated to: {args.set_default_model} in ~/.ab/config.json")
        else:
            pp("Error updating default model.")
        # If only set default and no prompt or paths provided, exit.
        if not args.prompt and len(args.paths) == 0:
            return

    # Load configurations
    config = load_config()
    settings = resolve_settings(args, config)

    path_format_option = 'full'
    if args.relative_paths:
        path_format_option = 'relative'
    elif args.filename_only:
        path_format_option = 'name_only'

    # Load .aiignore patterns
    aiignore_files = find_aiignore_files(pathlib.Path.cwd())
    aiignore_spec = load_aiignore_spec(aiignore_files)
    if aiignore_files:
        pp(f"Loaded .aiignore from: {', '.join(str(f) for f in aiignore_files)}")

    all_files_content = []
    total_word_count = 0
    total_estimated_tokens = 0
    files_processed_count = 0
    files_error_count = 0
    files_skipped_count = 0

    for path_arg in args.paths:
        if not path_arg.exists():
            pp(f"Warning: Path '{path_arg}' does not exist. Skipping.")
            continue

        base_path = path_arg.resolve() if path_arg.is_dir() else path_arg.parent.resolve()

        if path_arg.is_file():
            # Check .aiignore
            if should_ignore_path(path_arg.resolve(), aiignore_spec, base_path):
                pp(f"Ignored by .aiignore: {path_arg}")
                files_skipped_count += 1
                continue
            # Check if binary
            if is_binary_file(path_arg):
                pp(f"Ignored (binary): {path_arg}")
                files_skipped_count += 1
                continue
            # Process text file
            content, word_count, estimated_tokens = process_file(path_arg, path_format_option, args.max_tokens_doc)
            pp(f"Processing file: {path_arg.resolve()} ({word_count} words, ~{estimated_tokens} tokens)")
            if content.startswith("// error_processing_file"):
                files_error_count += 1
            else:
                files_processed_count += 1
                total_word_count += word_count
                total_estimated_tokens += estimated_tokens
            all_files_content.append(content)

        elif path_arg.is_dir():
            pp(f"Processing directory: {path_arg.resolve()}")
            for child_path in path_arg.rglob('*'):
                if child_path.is_file():
                    # Check .aiignore
                    if should_ignore_path(child_path.resolve(), aiignore_spec, base_path):
                        files_skipped_count += 1
                        continue
                    # Check if binary
                    if is_binary_file(child_path):
                        files_skipped_count += 1
                        continue
                    # Process text file
                    content, word_count, estimated_tokens = process_file(child_path, path_format_option, args.max_tokens_doc)
                    pp(f"  -> Processing: {child_path.relative_to(path_arg)} ({word_count} words, ~{estimated_tokens} tokens)")
                    if content.startswith("// error_processing_file"):
                        files_error_count += 1
                    else:
                        files_processed_count += 1
                        total_word_count += word_count
                        total_estimated_tokens += estimated_tokens
                    all_files_content.append(content)
        else:
            pp(f"Warning: Path '{path_arg}' is not a file or directory. Skipping.")

    final_text = "".join(all_files_content)

    # If no files were processed
    if not final_text and not args.prompt:
        pp("\nNo valid files were found or processed.")
        if files_skipped_count > 0:
            pp(f"{files_skipped_count} file(s) were ignored (binary or .aiignore).")
        return

    original_total_tokens = len(final_text) // 4
    if args.max_tokens and original_total_tokens > args.max_tokens:
        pp(f"\nWarning: Final context with ~{original_total_tokens} tokens exceeded limit of {args.max_tokens}. Truncating...")
        max_chars = args.max_tokens * 4
        final_text = final_text[:max_chars]
        pp(f"New estimated token count in context: ~{len(final_text) // 4}")

    # Make OpenRouter call if prompt exists
    if args.prompt:
        model = settings["model"]
        timeout_s = settings["timeout_s"]
        api_key_env = settings["api_key_env"]
        api_base = settings["api_base"]

        max_tokens = 0 if args.unlimited else args.max_completion_tokens
        result = send_to_openrouter(
            args.prompt, final_text, args.lang, args.specialist,
            model, timeout_s, max_tokens,
            api_key_env=api_key_env, api_base=api_base
        )

        if result:
            response_text = result['text']

            if VERBOSE:
                pp("\n--- REQUEST INFORMATION ---")
                pp(f"Provider Used: {result['provider']}")
                pp(f"Model Used: {result['model']}")
                pp(f"Files Processed: {files_processed_count} ({total_word_count} words, ~{total_estimated_tokens} tokens) | Errors: {files_error_count} | Ignored: {files_skipped_count}")
                pp(f"Tokens Sent (API): {result['prompt_tokens']}")
                pp(f"Tokens Received (API): {result['response_tokens']}")
                pp("---------------------------------")

                pp("\n--- MODEL RESPONSE ---\n")
                pp(response_text)
                pp("\n--------------------------\n")
            else:
                text = response_text.strip()

                if args.json:
                    if text.startswith('```json'):
                        text = text.replace('```json', '').replace('```', '')

                    try:
                        text = json.dumps(json.loads(text), indent=4)
                    except Exception:
                        pass

                print(text, flush=True)

            # Skip clipboard when not in verbose mode (subprocess calls)
            if VERBOSE:
                try:
                    pyperclip.copy(response_text)
                    pp("Response copied to clipboard!")
                except pyperclip.PyperclipException as e:
                    pp(f"Error: Could not copy to clipboard. {e}")

            # Prepare processed files information
            files_info = {
                'processed': files_processed_count,
                'errors': files_error_count,
                'skipped': files_skipped_count,
                'words': total_word_count,
                'tokens': total_estimated_tokens,
                'file_list': [str(p) for p in args.paths]
            }

            save_to_history(result['full_prompt'], response_text, result, files_info, args)
        else:
            # API call failed - exit with error code
            sys.exit(1)
        return

    # If no prompt but file content exists, copy to clipboard
    if final_text:
        try:
            pyperclip.copy(final_text)
            pp(f"\nProcessed {files_processed_count} file(s) successfully ({total_word_count} words, ~{total_estimated_tokens} tokens total).")
            if files_skipped_count > 0:
                 pp(f"{files_skipped_count} file(s) were ignored (binary or .aiignore).")
            if files_error_count > 0:
                pp(f"Found errors in {files_error_count} file(s).")
            pp("Combined content was copied to your clipboard!")
        except pyperclip.PyperclipException as e:
            pp(f"\nError: Could not copy to clipboard. {e}")
            pp("\nHere is the combined output:\n")
            pp("--------------------------------------------------")
            pp(final_text)
            pp("--------------------------------------------------")


if __name__ == "__main__":
    main()
