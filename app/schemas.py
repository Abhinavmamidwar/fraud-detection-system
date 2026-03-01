from pydantic import BaseModel, Field
from typing import List


class TransactionData(BaseModel):
    """Transaction data for fraud detection"""
    step: int = Field(..., example=1, description="Time step (1 = first hour)")
    type: str = Field(..., example="TRANSFER", description="Transaction type: PAYMENT, TRANSFER, CASH_OUT, CASH_IN, DEBIT")
    amount: float = Field(..., example=10000.0, description="Transaction amount")
    nameOrig: str = Field(..., example="C12345", description="Sender ID (C=Customer, M=Merchant)")
    oldbalanceOrg: float = Field(..., example=15000.0, description="Sender's balance before transaction")
    nameDest: str = Field(..., example="M67890", description="Receiver ID (C=Customer, M=Merchant)")

    class Config:
        schema_extra = {
            "example": {
                "step": 1,
                "type": "TRANSFER",
                "amount": 10000.0,
                "nameOrig": "C12345",
                "oldbalanceOrg": 15000.0,
                "nameDest": "M67890"
            }
        }


class PredictionResponse(BaseModel):
    """Fraud prediction response"""
    risk_score: float = Field(..., example=24.2, description="Risk score 0-100%")
    risk_level: str = Field(..., example="SAFE", description="SAFE, WATCH, FLAG, BLOCK")
    action: str = Field(..., example="ALLOW", description="Action to take")
    message: str = Field(..., example="✅ Transaction approved", description="Human-readable message")

    class Config:
        schema_extra = {
            "example": {
                "risk_score": 24.2,
                "risk_level": "SAFE",
                "action": "ALLOW",
                "message": "✅ Transaction approved"
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., example="healthy")

    class Config:
        schema_extra = {
            "example": {
                "status": "healthy"
            }
        }


class InfoResponse(BaseModel):
    """Model information response"""
    model_type: str
    thresholds: dict
    features_count: int
    features: List[str]

    class Config:
        schema_extra = {
            "example": {
                "model_type": "RandomForestClassifier",
                "thresholds": {
                    "BLOCK": ">20%",
                    "FLAG": "11-20%",
                    "WATCH": "9-11%",
                    "SAFE": "<9%"
                },
                "features_count": 20,
                "features": ["step", "amount", "oldbalanceOrg", "..."]
            }
        }


# Export all models
__all__ = [
    "TransactionData",
    "PredictionResponse",
    "HealthResponse",
    "InfoResponse"
]