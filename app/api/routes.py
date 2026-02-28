from fastapi import APIRouter, Request, HTTPException
from typing import List
from app.schemas import TransactionData, PredictionResponse
from app.services.predictor import predict_single, predict_batch

router = APIRouter()


@router.get("/")
async def home():
    return {"message": "Fraud Detection API", "version": "2.0.0"}


@router.get("/health")
async def health():
    return {"status": "healthy"}


@router.post("/predict", response_model=PredictionResponse)
async def predict(transaction: TransactionData, request: Request):
    try:
        model = request.app.state.model
        scaler = request.app.state.scaler
        feature_list = request.app.state.feature_list

        result = predict_single(
            model,
            scaler,
            feature_list,
            transaction.dict(),
        )

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/batch")
async def predict_multiple(
    transactions: List[TransactionData], request: Request
):
    try:
        model = request.app.state.model
        scaler = request.app.state.scaler
        feature_list = request.app.state.feature_list

        results = predict_batch(
            model,
            scaler,
            feature_list,
            [t.dict() for t in transactions],
        )

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
