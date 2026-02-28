import joblib
from app.core.config import MODEL_PATH, SCALER_PATH, FEATURE_PATH


def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_list = joblib.load(FEATURE_PATH)
    return model, scaler, feature_list
