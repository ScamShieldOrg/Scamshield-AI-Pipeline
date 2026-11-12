"""
==============================================================================
ScamShield-VN: AI Pipeline Microservice (FastAPI REST Service)
==============================================================================
Provides decoupled RESTful AI inference endpoints for:
- Spring Boot Backend Service (REST Client)
- Mobile App & Web Frontend (Direct / Dev testing)
- Automated Microservice Health & Monitoring
==============================================================================
"""

import os
import sys
import time
import uuid
import logging
import threading
import re
from datetime import datetime, timezone
from typing import Dict, Any, List

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Ensure root directory is on PYTHONPATH
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

load_dotenv(os.path.join(ROOT_DIR, ".env"))

from pipeline.two_tier_router import TwoTierRouter
from pipeline.verdict_contract import normalize_verdict
from pipeline.tier2_cache import Tier2CacheManager
from api.visobert_app import router as visobert_router
from pipeline.persistence import check_database, save_scan
from pipeline.rag import BgeM3Embedder, RagRetriever, apply_rag_migration
from pipeline.rag_ingestion import RagIngestor
from scripts.update_phishtank import feed_path, main as download_phishtank_feed
from api.schemas import (
    ScanRequest,
    StandardScanResponse,
    ScanResultData,
    Verdict,
    RadarMetrics,
    CouncilBreakdown,
    AgentFinding,
    HealthResponse,
)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ScamShield.API")

# FastAPI App setup
app = FastAPI(
    title="ScamShield-VN AI Pipeline Engine",
    description="2-Tier Cascaded AI Inference Service (Tier-1 ViSoBERT Fast-Path + Tier-2 3-Agent Forensic Council)",
    version="2.0.0",
)
# Direct model-inspection endpoint and /api/v1/scan share the same cached
# fine-tuned ViSoBERT checkpoint (models/visobert_finetuned/).
app.include_router(visobert_router)

# Cross-Origin Resource Sharing (CORS) for local frontend & Spring Boot
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router_instance: TwoTierRouter = None
cache_manager = Tier2CacheManager()
_phishtank_bootstrap_started = False


def _download_missing_phishtank_feed():
    """Report the outcome of the one-time background download in API logs."""
    if download_phishtank_feed() == 0:
        logger.info("PhishTank download completed; Tier 1 will use feed: %s", feed_path())
    else:
        logger.warning("PhishTank download failed; Tier 1 will report this source as unavailable until a feed exists. The next API startup will retry.")


@app.on_event("startup")
def startup_event():
    global router_instance, _phishtank_bootstrap_started

    if router_instance is not None:
        return
    logger.info("Initializing ScamShield TwoTierRouter...")
    try:
        rag_retriever = None
        if not hasattr(app.state, "rag_embedder"):
            app.state.rag_embedder = None
        if not hasattr(app.state, "rag_ingestor"):
            app.state.rag_ingestor = None
        if os.getenv("RAG_ENABLED", "true").lower() == "true":
            try:
                apply_rag_migration()
                if app.state.rag_embedder is None:
                    app.state.rag_embedder = BgeM3Embedder()
                rag_retriever = RagRetriever(app.state.rag_embedder)
                if app.state.rag_ingestor is None:
                    app.state.rag_ingestor = RagIngestor(app.state.rag_embedder)
                    app.state.rag_ingestor.start_daily_refresh()
                logger.info("RAG BGE-M3 loaded once; ingestion and inference share the instance.")
            except Exception as rag_error:
                logger.warning("RAG unavailable at startup: %s", type(rag_error).__name__)
        router_instance = TwoTierRouter(
            confidence_threshold=float(os.getenv("TIER1_CONFIDENCE_THRESHOLD", "0.90")),
            api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
            config_path=os.path.join(ROOT_DIR, "config", "gemini_models.yaml"),
            council_config_path=os.path.join(ROOT_DIR, "config", "council_registry.yaml"),
            use_council=os.getenv("USE_COUNCIL", "true").lower() == "true",
            rag_retriever=rag_retriever,
        )
        safe_browsing_key = router_instance.url_signals.safe_browsing.api_key
        if safe_browsing_key and not safe_browsing_key.lower().startswith("your_"):
            logger.info("Safe Browsing configured; API key will be verified on the first URL lookup.")
        else:
            logger.warning("Safe Browsing unavailable: SAFE_BROWSING_API_KEY is missing or a placeholder.")

        path = feed_path()
        if path.is_file():
            logger.info("PhishTank feed file found: %s (%d bytes); Tier 1 will use this snapshot.", path, path.stat().st_size)
        elif os.getenv("PHISHTANK_AUTO_DOWNLOAD", "true").lower() == "true" and not _phishtank_bootstrap_started:
            _phishtank_bootstrap_started = True
            logger.info("PhishTank feed missing at %s; starting one-time background download.", path)
            threading.Thread(target=_download_missing_phishtank_feed, name="phishtank-bootstrap", daemon=True).start()
        elif not _phishtank_bootstrap_started:
            logger.warning("PhishTank feed missing at %s; automatic download is disabled.", path)
        logger.info("ScamShield TwoTierRouter successfully initialized and ready for requests.")
    except Exception as e:
        logger.error(f"Failed to initialize TwoTierRouter: {e}", exc_info=True)


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint for Docker container & Kubernetes readiness probe."""
    tier1_ready = router_instance is not None and router_instance.tier1_classifier is not None
    tier2_ready = router_instance is not None and (router_instance.council is not None or router_instance.tier2_registry is not None)
    database_ready = check_database()
    return HealthResponse(
        status="healthy" if (tier1_ready and tier2_ready and database_ready) else "degraded",
        version="2.0.0",
        service="ScamShield AI Pipeline Engine",
        tier1_ready=tier1_ready,
        tier2_ready=tier2_ready,
        database_ready=database_ready,
    )


def _compute_radar_metrics(message: str, is_scam: bool, raw_res: Dict[str, Any]) -> RadarMetrics:
    """Computes explainable radar dimensions for frontend radar chart."""
    msg_lower = message.lower()

    urgency = 15
    authority = 15
    financial = 15
    channel = 15
    data_harvest = 10

    if any(k in msg_lower for k in ["khẩn", "ngay", "24h", "khoá", "bị khóa", "hết hạn", "trước khi"]):
        urgency += 55
    if any(k in msg_lower for k in ["công an", "viện kiểm sát", "bộ công an", "vietcombank", "mb bank", "thuế"]):
        authority += 60
    if any(k in msg_lower for k in ["trúng thưởng", "tri ân", "tặng quà", "hoa hồng", "việc làm", "vnd", "tiền"]):
        financial += 55
    if any(k in msg_lower for k in ["zalo", "telegram", "inbox", "link", "nhấp vào", "http", ".xyz", ".top"]):
        channel += 55
    if any(k in msg_lower for k in ["otp", "mật khẩu", "cccd", "số tài khoản", "sinh trắc", "đăng nhập", "xác minh"]):
        data_harvest += 60

    if not is_scam:
        urgency = min(urgency, 30)
        authority = min(authority, 30)
        financial = min(financial, 25)
        channel = min(channel, 20)
        data_harvest = min(data_harvest, 15)

    return RadarMetrics(
        urgency=urgency,
        authority_impersonation=authority,
        financial_lure=financial,
        suspicious_channel=channel,
        data_harvesting=data_harvest,
    )


def _build_recommendations(content_or_is_scam=None, is_scam=False, risk_level="LOW_RISK") -> List[str]:
    """
    GĐ4: Cung cấp khuyến nghị an toàn dựa trên tín hiệu có trong nội dung và chuẩn hóa hợp đồng.
    Hỗ trợ cả chữ ký _build_recommendations(content, is_scam, risk_level) lẫn _build_recommendations(is_scam, risk_level).
    """
    if isinstance(content_or_is_scam, bool):
        is_scam = content_or_is_scam
        content = ""
    else:
        content = str(content_or_is_scam or "")

    recs = []
    content_lower = content.lower()
    
    # 1. Khuyến nghị theo tín hiệu cụ thể trong tin nhắn (hỗ trợ cả có dấu và không dấu)
    has_financial = any(w in content_lower for w in ["chuyển", "chuyen", "tiền", "tien", "nạp", "nap", "vnd", "mượn", "muon", "thanh toán", "thanh toan", "cước", "cuoc", "stk", "tài khoản", "tai khoan"])
    has_link = bool(re.search(r"https?://[^\s]+", content_lower)) or any(w in content_lower for w in ["link", "truy cập", "truy cap", "nhấp", "nhap", "http", "www.", ".xyz", ".top", ".site"])
    has_auth = any(w in content_lower for w in ["otp", "mật khẩu", "mat khau", "mã xác", "ma xac", "cccd", "sinh trắc", "sinh trac", "đăng nhập", "dang nhap"])

    if has_financial:
        recs.append("Có yêu cầu liên quan đến tài chính: Tuyệt đối không chuyển tiền. Hãy gọi điện xác minh trực tiếp qua số bạn đã biết từ trước.")
    if has_link:
        recs.append("Có liên kết đính kèm: Tuyệt đối không nhấp vào đường link lạ. Hãy chủ động truy cập ứng dụng hoặc website chính thức.")
    if has_auth:
        recs.append("Có yếu tố mã xác thực: Không cung cấp mã OTP, mật khẩu hay thông tin định danh cho bất kỳ ai qua tin nhắn.")

    if not recs:
        if risk_level == "SCAM_RISK":
            recs = [
                "Khuyến nghị chưa vội chuyển tiền và không nhấp vào liên kết trong tin nhắn.",
                "Không cung cấp mật khẩu ngân hàng, mã OTP, số CCCD hoặc thông tin sinh trắc học.",
                "Nên gọi điện thoại trực tiếp vào số chính thức đã biết của đối phương để xác minh.",
            ]
        elif risk_level == "UNKNOWN":
            recs = [
                "Chưa đủ bằng chứng để kết luận; kiểm tra lại qua kênh chính thức hoặc nguồn độc lập.",
                "Không cung cấp OTP, mật khẩu hoặc thông tin tài chính khi chưa xác minh.",
            ]
        elif risk_level in ["SUSPICIOUS", "MEDIUM"]:
            recs = [
                "Tin nhắn có dấu hiệu bất thường, cần kiểm tra lại qua kênh liên hệ chính thống.",
                "Xác minh danh tính người gửi qua số điện thoại tổng đài chính thức của đơn vị liên quan.",
            ]
        else:
            recs = [
                "Tin nhắn không có dấu hiệu lừa đảo nguy hiểm.",
                "Tiếp tục cảnh giác với các yêu cầu chuyển tiền bất thường phát sinh sau này.",
            ]

    return recs


@app.post("/api/v1/scan", response_model=StandardScanResponse)
def scan_message(req: ScanRequest):
    """
    Main Scan Endpoint conforming to ScamShield Contract (docs/04_API_CONTRACTS_AND_SCHEMAS.md).
    SMS uses ViSoBERT plus URL signals; standalone URL uses URL signals only.
    """
    global router_instance
    if not router_instance:
        startup_event()
    if not router_instance:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI Pipeline Engine is still initializing. Please retry shortly.",
        )

    content_to_scan = req.content

    # GĐ5: Kiểm tra Cache Tier-2 trước khi gọi Cloud tốn kém
    cached_hit = None
    if not req.force_tier2:
        try:
            cached_hit = cache_manager.lookup(
                input_type="URL" if req.check_type == "URL" else "MESSAGE",
                content=content_to_scan,
            )
        except Exception as e:
            logger.warning(f"Cache lookup failed: {e}")

    if cached_hit:
        raw_res = cached_hit
    else:
        try:
            raw_res = router_instance.classify(content_to_scan, force_tier2=req.force_tier2, check_type=req.check_type)
        except Exception as e:
            logger.error(f"Inference error during scan: {e}", exc_info=True)
            return StandardScanResponse(
                status="error",
                error=f"Model inference failed: {str(e)}",
            )
        
        # GĐ5: Lưu kết quả Tier 2 thành công vào Cache
        if raw_res.get("routing_path") in ("TIER_2_FORENSIC_COUNCIL", "TIER_2_CLOUD_FALLBACK"):
            try:
                cache_manager.store(
                    input_type="URL" if req.check_type == "URL" else "MESSAGE",
                    content=content_to_scan,
                    raw_result=raw_res,
                )
            except Exception as e:
                logger.warning(f"Failed to store Tier-2 analysis in cache: {e}")

    is_scam = raw_res.get("is_scam", False)
    confidence = float(raw_res.get("final_confidence", 0.0))
    tier_executed = 2 if raw_res.get("escalated", False) else 1
    final_label = raw_res.get("final_label", "")
    risk_level = normalize_verdict(final_label)
    is_scam = risk_level == "SCAM_RISK"
    routing_path = raw_res.get("routing_path", "")
    model_score = round(confidence * 100.0, 1)

    if risk_level == "UNKNOWN":
        risk_score = None
    elif is_scam:
        # GĐ1: Bỏ hoàn toàn nâng điểm giả 88.5; giữ nguyên điểm thật của mô hình
        risk_score = model_score
    else:
        risk_score = round((1.0 - confidence) * 100.0, 1)

    # GĐ1: Xác định analysis_status trung thực
    if routing_path == "TIER_1_GRACEFUL_DEGRADATION":
        analysis_status = "TIER2_FAILED"
    elif final_label in {"REVIEW", "SUSPICIOUS_REVIEW", "UNKNOWN"}:
        analysis_status = "INSUFFICIENT_EVIDENCE"
    else:
        analysis_status = "OK"

    # GĐ1: Ràng buộc cảnh báo: Phân tích lỗi không bao giờ đi kèm warning_level = NONE
    if analysis_status in {"MODEL_UNAVAILABLE", "TIER2_FAILED"}:
        warning_level = "UNVERIFIED"
    elif risk_level == "DANGEROUS":
        warning_level = "HIGH"
    elif risk_level == "SUSPICIOUS" or final_label == "UNKNOWN":
        warning_level = "CAUTION"
    else:
        warning_level = "NONE"

    # GĐ1: Xác định decided_by engine
    if routing_path == "TIER_1_DIRECT_FASTPATH":
        decided_by = "TIER1"
    elif routing_path == "TIER_2_FORENSIC_COUNCIL":
        decided_by = "COUNCIL"
    elif routing_path == "TIER_2_CLOUD_FALLBACK":
        decided_by = "CLOUD"
    else:
        decided_by = "TIER1"

    # GĐ1: Trích xuất bằng chứng văn bản (evidence) có thật trong input
    evidence = []
    matched_cues = raw_res.get("tier1_telemetry", {}).get("matched_cues", [])
    if isinstance(matched_cues, list):
        for cue in matched_cues:
            if isinstance(cue, str) and cue.lower() in content_to_scan.lower() and cue not in evidence:
                evidence.append(cue)

    # GĐ1: Phân loại nội dung riêng (content_class)
    content_class = None
    lower_content = content_to_scan.lower()
    if final_label == "SCAM-RISK":
        content_class = "SCAM_ATTEMPT"
    elif any(kw in lower_content for kw in ["khuyen mai", "uu dai", "voucher", "giam gia", "tai app", "nohu", "game bai"]):
        content_class = "UNTRUSTED_PROMOTION"
    elif any(kw in lower_content for kw in ["ma xac minh", "otp", "mat khau"]):
        content_class = "AUTH_CREDENTIAL" 

    # Format Council Breakdown if Tier 2 executed
    council_breakdown = None
    if tier_executed == 2 and "tier2_telemetry" in raw_res:
        t2_telemetry = raw_res["tier2_telemetry"]
        specialist_votes = t2_telemetry.get("specialist_votes", {})
        votes_list = list(specialist_votes.values()) if isinstance(specialist_votes, dict) else specialist_votes
        agents_list = []
        for v in votes_list:
            v_vote = str(v.get("vote", "")).upper()
            verdict = normalize_verdict(v_vote)
            agents_list.append(
                AgentFinding(
                    role=v.get("display_name", v.get("role_id", "Specialist")),
                    verdict=verdict,
                    score=round(float(v.get("scam_prob", v.get("scam_probability", 0.0))) * 100.0, 1),
                    finding=v.get("analysis", v.get("rationale", "Phân tích chuyên sâu từ Chuyên gia AI")),
                )
            )

        scam_count = sum(1 for v in votes_list if "SCAM" in str(v.get("vote", "")).upper())
        total_votes = len(votes_list) if votes_list else 3
        council_breakdown = CouncilBreakdown(
            consensus_ratio=f"{scam_count}/{total_votes}",
            lead_model=raw_res.get("model_name", "ForensicCouncil-v2.0"),
            early_exit=t2_telemetry.get("early_exit", False),
            defender_intervened=t2_telemetry.get("defender_intervened", False),
            agents=agents_list,
        )

    scan_id = f"SC-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
    timestamp = datetime.now(timezone.utc).isoformat()

    verdict = Verdict(
        risk_level=risk_level,
        risk_score=risk_score,
        scam_category=raw_res.get("scam_type", "Chưa phân loại"),
        confidence=round(confidence, 4),
        summary=(raw_res.get("explanation") or "").strip() or "Chưa đủ bằng chứng để kết luận. Hãy xác minh qua kênh chính thức.",
        analysis_status=analysis_status,
        model_score=model_score,
        warning_level=warning_level,
        decided_by=decided_by,
        content_class=content_class,
        evidence=evidence,
    )

    radar = _compute_radar_metrics(content_to_scan, is_scam, raw_res)
    recommendations = _build_recommendations(content_to_scan, is_scam, risk_level)

    result_data = ScanResultData(
        scan_id=scan_id,
        timestamp=timestamp,
        tier_executed=tier_executed,
        verdict=verdict,
        radar_metrics=radar,
        council_breakdown=council_breakdown,
        actionable_recommendations=recommendations,
        telemetry={
            "routing_path": raw_res.get("routing_path"),
            "routing_reason": raw_res.get("routing_reason"),
            "total_latency_ms": raw_res.get("total_latency_ms"),
            "total_cost_usd": raw_res.get("total_cost_usd"),
            "tier1": raw_res.get("tier1_telemetry"),
            "tier2": raw_res.get("tier2_telemetry"),
            "final_label": raw_res.get("final_label"),
        },
    )

    try:
        verdict_str = risk_level
        save_scan(
            content=req.content,
            input_type=req.check_type,
            core_check_id=req.core_check_id,
            raw_result=raw_res,
            verdict=verdict_str,
            confidence=confidence,
            explanation=verdict.summary,
            guidance="\n".join(recommendations),
        )
    except Exception as e:
        logger.error("Failed to persist scan result: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference completed but the result could not be saved to the database.",
        ) from e

    return StandardScanResponse(
        status="success",
        data=result_data,
        error=None,
    )


@app.post("/api/v1/scan/raw")
def scan_message_raw(req: ScanRequest):
    """Debug endpoint returning exact internal router output dictionary."""
    if not router_instance:
        raise HTTPException(status_code=503, detail="AI Engine not initialized")
    return router_instance.classify(req.content, force_tier2=req.force_tier2, check_type=req.check_type)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app.py:app", host="0.0.0.0", port=8000, reload=True)
