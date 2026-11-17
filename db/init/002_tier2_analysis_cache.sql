-- ==============================================================================
-- ScamShield-VN: Tier 2 Forensic Council Analysis Cache (Phase 5 / GĐ5)
-- ==============================================================================
-- Stores successful Tier 2 analyses for reuse, preventing redundant cloud LLM calls.
-- Strict invariants:
--   1. Curated Blacklist/Whitelist always takes precedence over cache.
--   2. Only successful Tier 2 analyses are cached (never errors or graceful degradations).
--   3. Indicator hash must match for reuse to avoid indicator spoofing.
--   4. Bound by TTL (SMS: 14-30 days, URL: 1-7 days) and model/prompt versioning.
-- ==============================================================================

CREATE TABLE IF NOT EXISTS ai_analysis_cache (
    id BIGSERIAL PRIMARY KEY,
    input_type VARCHAR(20) NOT NULL,           -- MESSAGE | URL
    exact_hash CHAR(64) NOT NULL,              -- SHA-256 of normalized text
    fuzzy_key BIGINT,                          -- SimHash 64-bit for near-duplicate indexing
    indicator_hash CHAR(64),                   -- SHA-256 of extracted indicators (links, numbers)
    masked_text TEXT,                          -- PII-masked content
    verdict VARCHAR(30) NOT NULL,              -- SCAM_RISK | LOW_RISK | UNKNOWN
    risk_level VARCHAR(20) NOT NULL,           -- SAFE | SUSPICIOUS | DANGEROUS
    confidence DECIMAL(6,5) NOT NULL,
    routing_path VARCHAR(50) NOT NULL,         -- TIER_2_FORENSIC_COUNCIL | TIER_2_CLOUD_FALLBACK
    ai_snapshot JSONB NOT NULL,                -- scan_id, radar, council breakdown, recommendations
    model_version VARCHAR(50) NOT NULL,
    prompt_version VARCHAR(50) NOT NULL,
    policy_version VARCHAR(50) NOT NULL DEFAULT 'v1',
    hit_count INT DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    UNIQUE (input_type, exact_hash, model_version, prompt_version, policy_version)
);

CREATE INDEX IF NOT EXISTS ai_analysis_cache_lookup_idx 
    ON ai_analysis_cache (input_type, exact_hash, model_version, prompt_version, policy_version, expires_at);

CREATE INDEX IF NOT EXISTS ai_analysis_cache_fuzzy_idx 
    ON ai_analysis_cache (input_type, indicator_hash, expires_at);
