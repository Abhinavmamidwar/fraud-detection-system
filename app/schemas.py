from pydantic import BaseModel
from typing import List


class TransactionData(BaseModel):
    step: int
    type: str
    amount: float
    nameOrig: str
    oldbalanceOrg: float
    nameDest: str


class PredictionResponse(BaseModel):
    risk_score: float
    risk_level: str
    action: str
    message: str
