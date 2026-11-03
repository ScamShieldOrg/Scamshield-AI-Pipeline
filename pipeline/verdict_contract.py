"""Public three-label contract; uncertain or unrecognized decisions fail closed."""

def normalize_verdict(label: str) -> str:
    label = str(label).upper()
    if label in {"SCAM", "SCAM-RISK", "SCAM_RISK", "DANGEROUS"}:
        return "SCAM_RISK"
    if label in {"HAM", "LOW-RISK", "LOW_RISK", "SAFE"}:
        return "LOW_RISK"
    return "UNKNOWN"
