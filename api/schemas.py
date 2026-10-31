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
    verdict: Literal["SAFE", "SUSPICIOUS", "DANGEROUS", "UNKNOWN"]
    score: float
    finding: str


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
    risk_level: Literal["SAFE", "SUSPICIOUS", "DANGEROUS", "UNKNOWN"]
    risk_score: Optional[float] = Field(..., ge=0.0, le=100.0, description="Null when risk is UNKNOWN")
    scam_category: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    summary: str


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
