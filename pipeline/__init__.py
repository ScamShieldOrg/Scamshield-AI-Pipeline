"""
ScamShield-VN Pipeline Package
Real-time Vietnamese Scam Detection (2-Tier Cascaded Architecture)
"""
from .model_registry import ModelRegistry
from .local_classifier import LocalPhoBERTClassifier, LocalViSoBERTClassifier
from .two_tier_router import TwoTierRouter
from .forensic_council import ForensicCouncil

TrainedViSoBERTClassifier = LocalViSoBERTClassifier

__all__ = [
    "ModelRegistry",
    "LocalPhoBERTClassifier",
    "LocalViSoBERTClassifier",
    "TwoTierRouter",
    "ForensicCouncil",
    "TrainedViSoBERTClassifier",
]
