from fastapi import FastAPI
from app.api.routes import router
from app.services.model_loader import load_artifacts

app = FastAPI(title="Fraud Detection API")

model, scaler, feature_list = load_artifacts()

app.state.model = model
app.state.scaler = scaler
app.state.feature_list = feature_list

app.include_router(router)
