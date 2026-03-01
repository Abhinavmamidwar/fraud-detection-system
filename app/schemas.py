from pydantic import BaseModel
from typing import List


class Transaction(BaseModel):
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float
    type: str

    class Config:
        schema_extra = {
            "example": {
                "amount": 2500,
                "oldbalanceOrg": 5000,
                "newbalanceOrig": 2500,
                "oldbalanceDest": 10000,
                "newbalanceDest": 12500,
                "type": "TRANSFER"
            }
        }



class PredictionResponse(BaseModel):
    risk_score: float
    risk_level: str
    action: str
    message: str
