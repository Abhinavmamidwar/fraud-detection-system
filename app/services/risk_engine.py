from app.core.config import THRESHOLDS


def calculate_risk_score(model_probability: float) -> float:
    scale_factor = 1 / 0.30
    risk_score = min(model_probability * scale_factor, 1.0)
    return round(risk_score * 100, 1)


def get_risk_decision(probability: float):
    risk_score = calculate_risk_score(probability)

    if probability >= THRESHOLDS["BLOCK"]:
        return {
            "risk_level": "BLOCK",
            "action": "BLOCK_IMMEDIATELY",
            "message": "Transaction blocked - Critical risk",
            "risk_score": risk_score,
        }

    elif probability >= THRESHOLDS["FLAG"]:
        return {
            "risk_level": "FLAG",
            "action": "MANUAL_REVIEW",
            "message": "Transaction flagged for manual review",
            "risk_score": risk_score,
        }

    elif probability >= THRESHOLDS["WATCH"]:
        return {
            "risk_level": "WATCH",
            "action": "SILENT_MONITOR",
            "message": "Transaction under observation",
            "risk_score": risk_score,
        }

    else:
        return {
            "risk_level": "SAFE",
            "action": "ALLOW",
            "message": "Transaction approved",
            "risk_score": risk_score,
        }
