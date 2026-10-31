"""Tier 2 prompt logging is explicit and preserves the sent prompt."""

import json
import logging

from pipeline.prompt_logging import log_tier2_prompt


def test_tier2_prompt_logging_is_off_by_default(monkeypatch, caplog):
    monkeypatch.delenv("TIER2_LOG_PROMPTS", raising=False)
    with caplog.at_level(logging.INFO):
        log_tier2_prompt(logging.getLogger("test.prompts"), stage="council_maker",
                         system_prompt="system", user_prompt="SMS and RAG evidence")
    assert not caplog.records


def test_tier2_prompt_logging_records_exact_prompts(monkeypatch, caplog):
    monkeypatch.setenv("TIER2_LOG_PROMPTS", "true")
    user_prompt = "Tin nhắn đã che PII\n[chunk_id:42] Chứng cứ tiếng Việt"
    with caplog.at_level(logging.INFO):
        log_tier2_prompt(logging.getLogger("test.prompts"), stage="council_maker",
                         role="financial", model="test-model",
                         system_prompt="Hướng dẫn hệ thống", user_prompt=user_prompt)

    payload = json.loads(caplog.records[0].message.removeprefix("Tier 2 prompt: "))
    assert payload == {"stage": "council_maker", "role": "financial", "model": "test-model",
                       "system_prompt": "Hướng dẫn hệ thống", "user_prompt": user_prompt}
