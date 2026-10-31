"""Opt-in logging of the exact prompts sent to Tier 2 models."""

import json
import logging
import os


def log_tier2_prompt(
    logger: logging.Logger,
    *,
    stage: str,
    system_prompt: str,
    user_prompt: str,
    model: str | None = None,
    role: str | None = None,
) -> None:
    if os.getenv("TIER2_LOG_PROMPTS", "false").lower() != "true":
        return

    logger.info(
        "Tier 2 prompt: %s",
        json.dumps(
            {
                "stage": stage,
                "role": role,
                "model": model,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            },
            ensure_ascii=False,
        ),
    )
