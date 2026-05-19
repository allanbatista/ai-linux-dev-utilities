"""Shared OpenRouter LLM request settings."""
from __future__ import annotations

import argparse

REASONING_EFFORT_CHOICES = ("xhigh", "high", "medium", "low", "minimal", "none")
SERVICE_TIER_CHOICES = ("default", "flex", "priority")

DEFAULT_REASONING_EFFORT = "medium"
DEFAULT_SERVICE_TIER = "default"


def add_llm_request_arguments(parser: argparse.ArgumentParser) -> None:
    """Add OpenRouter override flags to a parser."""
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        choices=REASONING_EFFORT_CHOICES,
        default=None,
        help="Override OpenRouter reasoning effort (default: config value)",
    )
    parser.add_argument(
        "--service-tier",
        type=str,
        choices=SERVICE_TIER_CHOICES,
        default=None,
        help="Override OpenRouter service tier (default: config value)",
    )
