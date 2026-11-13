"""
==============================================================================
ScamShield-VN: Pydantic Data Contracts for AI Pipeline Microservice
==============================================================================
Complies with docs/04_API_CONTRACTS_AND_SCHEMAS.md for Frontend React & Spring Boot.
==============================================================================
"""

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator


class ScanRequest(BaseModel):
    """Payload sent by Core for an SMS or standalone URL check."""
    content: str = Field(..., min_length=1, description="Content to scan")
    sender: Optional[str] = Field(None, description="Phone number, email or brand header")
    check_type: Literal["SMS", "URL"] = Field(..., description="SMS text or standalone URL; Core handles phone and bank account checks")
    target_bank: Optional[str] = Field(None, description="Bank code (e.g. VCB, MB, BIDV)")
    target_account_number: Optional[str] = Field(None, description="Bank account number if check_type is BANK_ACCOUNT")
    force_tier2: bool = Field(False, description="Force Tier-2 Council review for SMS regardless of Tier 1 confidence")
    core_check_id: Optional[int] = Field(None, gt=0, description="Core check_requests.id used as the idempotency key")

    @field_validator("content")
    @classmethod
    def nonblank_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("content must contain non-whitespace text")
        return value


class AgentFinding(BaseModel):
    role: str
    verdict: Literal["SCAM_RISK", "LOW_RISK", "UNKNOWN"]
    score: float
    finding: str
    cited_chunk_ids: List[int] = Field(default_factory=list)


class CouncilBreakdown(BaseModel):
    consensus_ratio: str
    lead_model: str
    early_exit: bool = False
    defender_intervened: bool = False
    agents: List[AgentFinding] = []


class RadarMetrics(BaseModel):
    urgency: int = Field(0, ge=0, le=100)
    authority_impersonation: int = Field(0, ge=0, le=100)
    financial_lure: int = Field(0, ge=0, le=100)
    suspicious_channel: int = Field(0, ge=0, le=100)
    data_harvesting: int = Field(0, ge=0, le=100)


class Verdict(BaseModel):
    risk_level: Literal["SCAM_RISK", "LOW_RISK", "UNKNOWN"]
    risk_score: Optional[float] = Field(..., ge=0.0, le=100.0, description="Null when risk is UNKNOWN")
    scam_category: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    summary: str
    analysis_status: Literal["OK", "MODEL_UNAVAILABLE", "TIER2_FAILED", "INSUFFICIENT_EVIDENCE"] = Field("OK", description="Status of model analysis")
    model_score: Optional[float] = Field(None, description="Raw model confidence score (0-100) before policy mapping")
    warning_level: Literal["NONE", "CAUTION", "HIGH", "UNVERIFIED"] = Field("NONE", description="Actionable warning level for end user")
    decided_by: Optional[Literal["BLACKLIST", "WHITELIST", "TIER1", "COUNCIL", "CLOUD", "CACHE", "RULES"]] = Field(None, description="Engine or tier that made decision")
    content_class: Optional[str] = Field(None, description="Specific content classification (e.g. UNTRUSTED_PROMOTION, AUTH_OTP)")
    evidence: List[str] = Field(default_factory=list, description="Verified textual evidence quotes from message")


class ScanResultData(BaseModel):
    scan_id: str
    timestamp: str
    tier_executed: int = Field(..., description="1: ViSoBERT On-Device/CPU | 2: Cloud 3-Agent Council")
    verdict: Verdict
    radar_metrics: RadarMetrics
    council_breakdown: Optional[CouncilBreakdown] = None
    actionable_recommendations: List[str] = []
    telemetry: Optional[Dict[str, Any]] = None


class StandardScanResponse(BaseModel):
    status: Literal["success", "error"]
    data: Optional[ScanResultData] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    service: str
    tier1_ready: bool
    tier2_ready: bool
    database_ready: bool
