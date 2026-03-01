from pydantic import BaseModel, Field
from typing import List


class Transaction(BaseModel):
    amount: float = Field(..., example=2500)
    oldbalanceOrg: float = Field(..., example=5000)
    newbalanceOrig: float = Field(..., example=2500)
    oldbalanceDest: float = Field(..., example=10000)
    newbalanceDest: float = Field(..., example=12500)
    type: str = Field(..., example="TRANSFER")


class PredictionResponse(BaseModel):
    risk_score: float
    risk_level: str
    action: str
    message: str
