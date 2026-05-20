"""History management utilities for ab-cli.

This module handles saving interaction history with LLMs,
including sanitization of sensitive data before storage.
"""
import argparse
import datetime
import hashlib
import json
import pathlib
import re
from typing import Any, Dict, Optional

from ab_cli.core.config import get_config
from ab_cli.utils.api import pp


def sanitize_sensitive_data(text: str) -> str:
    """
    Sanitize sensitive data from text before saving to history.

    Patterns sanitized:
    - API keys (various formats including custom X_API_KEY patterns)
    - Passwords and secrets
    - Tokens and credentials (OAuth, Bearer, access tokens)
    - Webhook URLs
    - Private keys (PEM format)
    - Generic secret patterns

    Args:
        text: The text to sanitize

    Returns:
        Text with sensitive data replaced with [REDACTED] placeholders
    """
    if not text:
        return text

    # Patterns to sanitize (key=value format)
    patterns = [
        # API keys - specific patterns
        (r'(api[_-]?key\s*[=:]\s*)["\']?[\w-]{20,}["\']?', r'\1[REDACTED]'),
        (r'(OPENROUTER_API_KEY\s*[=:]\s*)["\']?[\w-]+["\']?', r'\1[REDACTED]'),
        # OpenAI-style sk- API keys (allow hyphens in key value)
        (r'(sk-[a-zA-Z0-9-]{20,})', '[REDACTED_API_KEY]'),
        # Custom API keys (e.g., STRIPE_API_KEY=xxx, GITHUB_API_KEY=xxx)
        (r'([A-Z_]+_API_KEY\s*[=:]\s*)\S+', r'\1[REDACTED]'),
        # Passwords
        (r'(password\s*[=:]\s*)["\']?[^\s"\']+["\']?', r'\1[REDACTED]', re.IGNORECASE),
        (r'(passwd\s*[=:]\s*)["\']?[^\s"\']+["\']?', r'\1[REDACTED]', re.IGNORECASE),
        (r'(pwd\s*[=:]\s*)["\']?[^\s"\']+["\']?', r'\1[REDACTED]', re.IGNORECASE),
        # Tokens and secrets
        (r'(secret\s*[=:]\s*)["\']?[\w-]+["\']?', r'\1[REDACTED]', re.IGNORECASE),
        (r'(token\s*[=:]\s*)["\']?[\w-]{20,}["\']?', r'\1[REDACTED]', re.IGNORECASE),
        (r'(auth\s*[=:]\s*)["\']?[\w-]+["\']?', r'\1[REDACTED]', re.IGNORECASE),
        # OAuth and access tokens
        (r'(oauth_token\s*[=:]\s*)\S+', r'\1[REDACTED]', re.IGNORECASE),
        (r'(access_token\s*[=:]\s*)\S+', r'\1[REDACTED]', re.IGNORECASE),
        # Bearer tokens (comprehensive pattern including base64 chars)
        (r'(Bearer\s+)[A-Za-z0-9\-._~+/]+=*', r'\1[REDACTED]'),
        # Basic auth
        (r'(Basic\s+)[a-zA-Z0-9+/=]{20,}', r'\1[REDACTED]'),
        # Webhook URLs (sanitize entire URL - matches "webhook" or "hooks" in URL)
        (r'https?://[^\s]*(webhook|hooks)[^\s]*', '[REDACTED_WEBHOOK_URL]', re.IGNORECASE),
        # Private keys (PEM format)
        (r'-----BEGIN[A-Z\s]*PRIVATE KEY-----[\s\S]*?-----END[A-Z\s]*PRIVATE KEY-----',
         '[REDACTED_PRIVATE_KEY]'),
        # Generic secrets pattern (SECRET, PASSWORD, TOKEN, KEY, CREDENTIAL in env vars)
        (r'([A-Z_]*(SECRET|PASSWORD|TOKEN|KEY|CREDENTIAL)[A-Z_]*\s*[=:]\s*)\S+',
         r'\1[REDACTED]'),
    ]

    result = text
    for pattern_tuple in patterns:
        if len(pattern_tuple) == 3:
            pattern, replacement, flags = pattern_tuple
            result = re.sub(pattern, replacement, result, flags=flags)
        else:
            pattern, replacement = pattern_tuple
            result = re.sub(pattern, replacement, result)

    return result


def calculate_estimated_cost(model: str, prompt_tokens: int, response_tokens: int) -> float:
    """
    Calculate estimated cost based on model and tokens used.

    Args:
        model: The model name/identifier
        prompt_tokens: Number of prompt tokens
        response_tokens: Number of response tokens

    Returns:
        Estimated cost in USD (approximate values)
    """
    if not isinstance(prompt_tokens, int) or not isinstance(response_tokens, int):
        return 0.0

    # Approximate prices per 1M tokens (USD) - update as needed
    pricing = {
        # OpenAI
        'gpt-4o': {'prompt': 2.50, 'response': 10.00},
        'gpt-4o-mini': {'prompt': 0.15, 'response': 0.60},
        'gpt-4-turbo': {'prompt': 10.00, 'response': 30.00},
        'gpt-4': {'prompt': 30.00, 'response': 60.00},
        'gpt-3.5-turbo': {'prompt': 0.50, 'response': 1.50},

        # Google Gemini (estimates)
        'gemini-1.5-pro': {'prompt': 3.50, 'response': 10.50},
        'gemini-1.5-flash': {'prompt': 0.075, 'response': 0.30},
        'gemini-pro': {'prompt': 0.50, 'response': 1.50},
    }

    # Find model price
    model_lower = model.lower()
    price_info = None

    for model_key, prices in pricing.items():
        if model_key in model_lower:
            price_info = prices
            break

    if not price_info:
        return 0.0

    # Calculate cost
    prompt_cost = (prompt_tokens / 1_000_000) * price_info['prompt']
    response_cost = (response_tokens / 1_000_000) * price_info['response']

    return round(prompt_cost + response_cost, 6)


def update_history_index(history_dir: pathlib.Path, entry: Dict[str, Any]) -> None:
    """
    Update the master index file with interaction summary.

    Args:
        history_dir: Path to the history directory
        entry: The history entry to add to the index
    """
    index_file = history_dir / "index.json"

    try:
        if index_file.exists():
            with open(index_file, 'r', encoding='utf-8') as f:
                index = json.load(f)
        else:
            index = {
                "created_at": datetime.datetime.now().isoformat(),
                "total_interactions": 0,
                "total_tokens_used": 0,
                "total_estimated_cost": 0.0,
                "interactions": []
            }

        # Add interaction summary
        summary = {
            "session_id": entry['metadata']['session_id'],
            "timestamp": entry['metadata']['timestamp'],
            "provider": entry['provider_info']['provider'],
            "model": entry['provider_info']['model'],
            "tokens": entry['tokens'].get('total_tokens', 'N/A'),
            "cost": entry['tokens'].get('estimated_cost_usd', 0.0),
            "files_processed": entry['files_info']['processed_count'],
            "response_preview": entry['content']['response']['preview']
        }

        index['interactions'].insert(0, summary)  # Most recent first
        index['total_interactions'] = len(index['interactions'])

        # Update totals
        if isinstance(entry['tokens'].get('total_tokens'), int):
            index['total_tokens_used'] += entry['tokens']['total_tokens']

        if isinstance(entry['tokens'].get('estimated_cost_usd'), (int, float)):
            index['total_estimated_cost'] += entry['tokens']['estimated_cost_usd']
            index['total_estimated_cost'] = round(index['total_estimated_cost'], 6)

        # Save index
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2, ensure_ascii=False)

    except Exception as e:
        pp(f"Warning: Could not update index: {e}")


def cleanup_old_history(history_dir: pathlib.Path, keep_last: int = 100) -> None:
    """
    Remove old history files, keeping only the last N.

    Args:
        history_dir: Path to the history directory
        keep_last: Number of history files to keep (default: 100)
    """
    try:
        history_files = sorted(
            history_dir.glob("history_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        if len(history_files) > keep_last:
            for old_file in history_files[keep_last:]:
                old_file.unlink()

    except Exception:
        # Non-critical, silent log
        pass


def save_to_history(full_prompt: str, response_text: str, result: Dict[str, Any],
                    files_info: Dict[str, Any], args: argparse.Namespace) -> None:
    """
    Save full interaction history with LLM to ~/.ab/history/

    Respects config history.enabled setting.
    Sanitizes sensitive data before saving.

    Information saved:
    - Request timestamp
    - Provider and model used
    - Full prompt and response (sanitized)
    - Token metrics (prompt, response, total)
    - Processed files information
    - Configuration used (specialist, language, etc)
    - Prompt hash to avoid duplicates

    Args:
        full_prompt: The full prompt sent to the LLM
        response_text: The response received from the LLM
        result: API result dictionary with metadata
        files_info: Information about processed files
        args: Command-line arguments namespace
    """
    try:
        # Check if history is enabled
        config = get_config()
        if not config.get_with_default('history.enabled'):
            return

        # Sanitize sensitive data
        sanitized_prompt = sanitize_sensitive_data(full_prompt)
        sanitized_response = sanitize_sensitive_data(response_text)

        # History directory
        history_dir = pathlib.Path.home() / ".ab" / "history"
        history_dir.mkdir(parents=True, exist_ok=True)

        # Filename based on timestamp
        timestamp = datetime.datetime.now()
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")

        # Prompt hash for unique reference
        prompt_hash = hashlib.md5(full_prompt.encode('utf-8')).hexdigest()[:8]

        # Full data structure
        history_entry = {
            "metadata": {
                "timestamp": timestamp.isoformat(),
                "timestamp_formatted": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "prompt_hash": prompt_hash,
                "session_id": f"{timestamp_str}_{prompt_hash}"
            },
            "provider_info": {
                "provider": result.get('provider', 'unknown'),
                "model": result.get('model', 'unknown'),
                "api_version": result.get('api_version', 'N/A')
            },
            "tokens": {
                "prompt_tokens": result.get('prompt_tokens', 'N/A'),
                "response_tokens": result.get('response_tokens', 'N/A'),
                "total_tokens": (
                    result.get('prompt_tokens', 0) + result.get('response_tokens', 0)
                    if isinstance(result.get('prompt_tokens'), int) and isinstance(result.get('response_tokens'), int)
                    else 'N/A'
                ),
                "estimated_cost_usd": calculate_estimated_cost(
                    result.get('model', ''),
                    result.get('prompt_tokens', 0),
                    result.get('response_tokens', 0)
                )
            },
            "files_info": {
                "processed_count": files_info.get('processed', 0),
                "error_count": files_info.get('errors', 0),
                "skipped_count": files_info.get('skipped', 0),
                "total_words": files_info.get('words', 0),
                "total_estimated_tokens": files_info.get('tokens', 0),
                "file_list": files_info.get('file_list', [])
            },
            "configuration": {
                "specialist": args.specialist if hasattr(args, 'specialist') else None,
                "language": args.lang if hasattr(args, 'lang') else 'en',
                "max_tokens": args.max_tokens if hasattr(args, 'max_tokens') else None,
                "max_tokens_doc": args.max_tokens_doc if hasattr(args, 'max_tokens_doc') else None,
                "max_completion_tokens": 0 if getattr(args, 'unlimited', False) else (args.max_completion_tokens if hasattr(args, 'max_completion_tokens') else 16000),
                "path_format": (
                    'relative' if args.relative_paths else
                    'name_only' if args.filename_only else
                    'full'
                ) if hasattr(args, 'relative_paths') else 'full'
            },
            "content": {
                "prompt": {
                    "full": sanitized_prompt,
                    "length_chars": len(sanitized_prompt),
                    "length_words": len(sanitized_prompt.split())
                },
                "response": {
                    "full": sanitized_response,
                    "length_chars": len(sanitized_response),
                    "length_words": len(sanitized_response.split()),
                    "preview": sanitized_response[:500] + "..." if len(sanitized_response) > 500 else sanitized_response
                }
            },
            "statistics": {
                "prompt_to_response_ratio": round(len(sanitized_response) / len(sanitized_prompt), 2) if sanitized_prompt else 0,
                "avg_response_word_length": round(len(sanitized_response) / max(len(sanitized_response.split()), 1), 2),
                "response_lines": sanitized_response.count('\n') + 1
            }
        }

        # Save individual file
        history_file = history_dir / f"history_{timestamp_str}_{prompt_hash}.json"
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history_entry, f, indent=2, ensure_ascii=False)

        # Update master index
        update_history_index(history_dir, history_entry)

        pp(f"History saved: {history_file}")

    except Exception as e:
        pp(f"Warning: Could not save history: {e}")
