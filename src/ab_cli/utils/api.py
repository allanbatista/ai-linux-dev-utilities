"""API utilities for ab-cli.

This module provides functions for communicating with LLM APIs.
Extracted from commands/prompt.py to avoid circular imports.
"""
import os
import sys
from typing import Any, Dict, Optional

import requests


# Module-level verbose flag (can be set by callers)
VERBOSE = True


def pp(*args, **kwargs):
    """Print only if VERBOSE is True."""
    if VERBOSE:
        print(*args, **kwargs)


def build_specialist_prefix(specialist: Optional[str]) -> str:
    """Build a specialist prompt prefix based on the given persona.

    Args:
        specialist: The specialist persona ('dev' or 'rm')

    Returns:
        The specialist prompt prefix string, or empty string if no specialist
    """
    specialist_prompts = {
        'dev': 'Act as a senior programmer specialized in software development, with over 20 years of experience. Your responses should be clear, efficient, well-structured and follow industry best practices. Think step by step.',
        'rm': 'Act as a senior Retail Media analyst, specialized in digital advertising strategies for e-commerce and marketplaces. Your knowledge covers platforms like Amazon Ads, Mercado Ads and Criteo. Your responses should be analytical, strategic and data-driven.'
    }
    return specialist_prompts.get(specialist or "", "")


def send_to_openrouter(prompt: str, context: str, lang: str, specialist: Optional[str],
                       model_name: str, timeout_s: int, max_completion_tokens: int = 256,
                       api_key_env: str = "OPENROUTER_API_KEY",
                       api_base: str = "https://openrouter.ai/api/v1") -> Optional[Dict[str, Any]]:
    """
    Sends the prompt and context to the OpenRouter API (OpenAI compatible).

    Args:
        prompt: The prompt text to send
        context: Additional context (file contents, etc.)
        lang: Output language code (e.g., 'en', 'pt-br')
        specialist: Optional specialist persona ('dev' or 'rm')
        model_name: The model identifier to use
        timeout_s: Request timeout in seconds
        max_completion_tokens: Maximum tokens for response (0 for unlimited)
        api_key_env: Environment variable name containing API key
        api_base: Base URL for the API

    Returns:
        Dict with response data or None on failure
    """
    api_key = os.getenv(api_key_env)
    if not api_key:
        # Always print error to stderr, regardless of VERBOSE
        print(f"Error: The environment variable {api_key_env} is not defined.", file=sys.stderr)
        return None

    # Build full prompt
    parts = []
    specialist_prefix = build_specialist_prefix(specialist)
    if specialist_prefix:
        parts.append(specialist_prefix)

    parts.append(prompt)

    if context.strip():
        parts.append("\n--- FILE CONTEXT ---\n" + context)

    parts.append(f"--- OUTPUT INSTRUCTION ---\nRespond strictly in language: {lang}.")

    full_prompt = "\n\n".join(parts)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    url = f"{api_base.rstrip('/')}/chat/completions"

    messages = [{"role": "user", "content": full_prompt}]
    if specialist_prefix:
        messages.insert(0, {"role": "system", "content": specialist_prefix})

    payload = {
        "model": model_name,
        "messages": messages,
    }

    if max_completion_tokens > 0:
        payload["max_tokens"] = max_completion_tokens

    try:
        pp(f"Sending request to OpenRouter ({model_name})...")
        response = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
        response.raise_for_status()
        data = response.json()

        message = data['choices'][0]['message']
        text_response = message.get('content') or ''

        # Handle reasoning models (gpt-5, o1, o3, etc.) that put response in reasoning field
        if not text_response and 'reasoning' in message:
            # For simple tasks, try to extract the final answer from reasoning
            reasoning = message.get('reasoning', '')
            # If the model ran out of tokens, reasoning might contain a partial answer
            if reasoning:
                pp(f"Note: Using reasoning field (model: {model_name}, content was empty)")
                text_response = reasoning

        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", "N/A")
        response_tokens = usage.get("completion_tokens", "N/A")

        return {
            "provider": "openrouter",
            "model": model_name,
            "text": text_response,
            "prompt_tokens": prompt_tokens,
            "response_tokens": response_tokens,
            "full_prompt": full_prompt,
        }

    except requests.exceptions.RequestException as e:
        # Always print errors to stderr regardless of VERBOSE mode
        print(f"Network or HTTP error calling OpenRouter: {e}", file=sys.stderr)
        if getattr(e, 'response', None) is not None:
            try:
                print(f"Error details: {e.response.text}", file=sys.stderr)
            except Exception:
                pass
        return None
    except (KeyError, IndexError) as e:
        print(f"Error extracting content from response: {e}", file=sys.stderr)
        try:
            print(f"Response structure received: {response.json()}", file=sys.stderr)
        except Exception:
            pass
        return None
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return None


def set_verbose(verbose: bool) -> None:
    """Set the module's verbose flag.

    Args:
        verbose: Whether to enable verbose output
    """
    global VERBOSE
    VERBOSE = verbose
